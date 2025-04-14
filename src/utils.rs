use bip0039::{Count, English, Mnemonic};
use hex::encode;
use secp256k1::{PublicKey, SecretKey};
use sha2::{Digest, Sha512};
use std::convert::TryInto;
use tiny_hderive::bip32::ExtendedPrivKey;
use tiny_keccak::{Hasher, Keccak};

// --- Precompute HMAC for speed ---
struct PrecomputeHmacSha512 {
    inner_key: [u8; 128],
    outer_key: [u8; 128],
}

impl PrecomputeHmacSha512 {
    fn new(key: &[u8]) -> Self {
        const BLOCK_SIZE: usize = 128; // for SHA-512
        let mut key_block = [0u8; BLOCK_SIZE];

        // If key is longer than block size, hash it, then pad.
        if key.len() > BLOCK_SIZE {
            let mut hasher = Sha512::new();
            hasher.input(key);
            let hashed = hasher.result();
            key_block[..hashed.len()].copy_from_slice(&hashed);
        } else {
            key_block[..key.len()].copy_from_slice(key);
        }

        let mut inner_key = [0u8; BLOCK_SIZE];
        let mut outer_key = [0u8; BLOCK_SIZE];

        for i in 0..BLOCK_SIZE {
            inner_key[i] = key_block[i] ^ 0x36;
            outer_key[i] = key_block[i] ^ 0x5c;
        }

        PrecomputeHmacSha512 {
            inner_key,
            outer_key,
        }
    }

    fn compute(&self, message: &[u8]) -> [u8; 64] {
        let mut hasher = Sha512::new();
        // inner-hash: inner_key || message
        hasher.input(&self.inner_key);
        hasher.input(message);
        let inner_result = hasher.result_reset();

        // outer-hash: outer_key || inner_result
        hasher.input(&self.outer_key);
        hasher.input(&inner_result);
        let outer_result = hasher.result();

        let mut result = [0u8; 64];
        result.copy_from_slice(&outer_result[..]);
        result
    }
}

// --- Holds all precomputed data for the "m/44'/60'/0'/0" parent key ---
struct PrecomputedParentForChild {
    parent_chain_code: [u8; 32],
    parent_secret_key: SecretKey,
    parent_public_key: [u8; 33],
    precomputed_hmac_sha512: PrecomputeHmacSha512,
}

// Safely extract chain_code from ExtendedPrivKey (which is 32 bytes after the secret key).
fn extract_chain_code(xprv: &ExtendedPrivKey) -> &[u8; 32] {
    // Unsafe cast for brevity – you can store chain_code in your own struct if you prefer
    unsafe {
        // Cast the reference to a raw pointer of type u8
        let raw_ptr = xprv as *const ExtendedPrivKey as *const u8;
        // Offset by 32 bytes (the size of secret_key) to reach chain_code
        let chain_code_ptr = raw_ptr.add(32) as *const [u8; 32];
        &*chain_code_ptr
    }
}

/// Create a `PrecomputedParentForChild` for the path m/44'/60'/0'/0 using the given mnemonic.
fn create_precomputed_parent_for_child(mnemonic: &Mnemonic<English>) -> PrecomputedParentForChild {
    let seed = mnemonic.to_seed("");

    // Derive the parent key at path "m/44'/60'/0'/0"
    let hdwallet = ExtendedPrivKey::derive(&seed, "m/44'/60'/0'/0")
        .expect("failed to derive path m/44'/60'/0'/0");

    // Our parent's secret & chain_code
    let secret_key =
        SecretKey::parse_slice(&hdwallet.secret()).expect("failed to parse parent's secret key");
    let parent_chain_code = *extract_chain_code(&hdwallet);

    // Compressed pubkey from the parent
    let parent_public_key =
        secp256k1::PublicKey::from_secret_key(&secret_key).serialize_compressed();

    // Precompute HMAC for all subsequent child derivation
    let precomputed_hmac_sha512 = PrecomputeHmacSha512::new(&parent_chain_code);

    PrecomputedParentForChild {
        parent_chain_code,
        parent_secret_key: secret_key,
        parent_public_key,
        precomputed_hmac_sha512,
    }
}

/// Quickly derive the secp256k1 public key for the given child index
/// (non-hardened only).
fn derive_child_public_key(parent: &PrecomputedParentForChild, child_index: u32) -> PublicKey {
    // HMAC( parent_public_key || child_index )
    let mut message = Vec::with_capacity(parent.parent_public_key.len() + 4);
    message.extend_from_slice(&parent.parent_public_key);
    message.extend_from_slice(&child_index.to_be_bytes());
    let result = parent.precomputed_hmac_sha512.compute(&message);

    // The left 32 bytes becomes the tweak
    let mut tweak = [0u8; 32];
    tweak.copy_from_slice(&result[..32]);

    let mut child_secret = SecretKey::parse_slice(&tweak).expect("invalid child secret");
    // add parent secret
    child_secret
        .tweak_add_assign(&parent.parent_secret_key)
        .expect("tweak_add_assign failed");
    // get public
    PublicKey::from_secret_key(&child_secret)
}

/// Perform Keccak-256 in place.
pub fn keccak_hash_in_place(input: &[u8], output: &mut [u8; 32]) {
    let mut hasher = Keccak::v256();
    hasher.update(input);
    hasher.finalize(output);
}

/// Perform SHA-512 in place.
pub fn sha512_hash_in_place(input: &[u8], output: &mut [u8; 64]) {
    let mut hasher = Sha512::new();
    hasher.input(input);
    let result = hasher.result();
    output.copy_from_slice(&result);
}

/// Perform secp256k1 scalar multiplication in place.
pub fn secp256k1_compute_pubkey_in_place(
    input: [u32; 8],
    output_px: &mut [u8; 32],
    output_py: &mut [u8; 32],
) {
    let secret_key = input
        .iter()
        .map(|x| x.to_le_bytes())
        .flatten()
        .collect::<Vec<u8>>();

    let secret_key = secp256k1::SecretKey::parse_slice(&secret_key).unwrap();
    let pub_key = secp256k1::PublicKey::from_secret_key(&secret_key);
    let pub_key_compressed = pub_key.serialize();
    output_px.copy_from_slice(&pub_key_compressed[1..33]);
    output_py.copy_from_slice(&pub_key_compressed[33..65]);
}

/// Convert 20-byte hex string to "0x"-prefixed EIP-55 checksum address.
fn checksum_address(address: &str) -> String {
    let address = address.trim_start_matches("0x").to_lowercase();
    // keccak hash of the lowercased hex
    let mut output = [0u8; 32];
    {
        let mut hasher = Keccak::v256();
        hasher.update(address.as_bytes());
        hasher.finalize(&mut output);
    }

    let hash_hex = encode(output);
    let mut result = String::with_capacity(42);
    result.push_str("0x");

    // For each nibble, if top 4 bits >= 8 => uppercase, else lowercase
    for (i, c) in address.chars().enumerate() {
        let hash_char = hash_hex.chars().nth(i).unwrap();
        if hash_char >= '8' {
            result.push(c.to_ascii_uppercase());
        } else {
            result.push(c);
        }
    }
    result
}

/// Simple function to return the Ethereum address for a given Mnemonic and child index.
/// Uses the precomputation approach from your code snippet for speed.
pub fn eth_address_from_mnemonic(mnemonic: &Mnemonic<English>, child_index: u32) -> String {
    // 1) Build the parent precomputation at "m/44'/60'/0'/0"
    let parent = create_precomputed_parent_for_child(mnemonic);

    // 2) Derive the child's public key
    let pub_key = derive_child_public_key(&parent, child_index);
    // 3) Keccak hash the last 64 bytes of the uncompressed pubkey (the x and y coords).
    //    The uncompressed pubkey is 65 bytes: [0x04, 32-byte X, 32-byte Y].
    //    Here, since we have a compressed 33-byte key, first decompress it:
    let uncompressed = pub_key.serialize();
    // skip the first byte 0x04
    let xyz = &uncompressed[1..];
    let mut keccak_out = [0u8; 32];
    keccak_hash_in_place(xyz, &mut keccak_out);

    // 4) Take the last 20 bytes => address
    let raw_addr = encode(&keccak_out[12..]); // 20 bytes
    // 5) Checksum it
    checksum_address(&raw_addr)
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    #[test]
    fn test_address_from_mnemonic() {
        let mnemonic = Mnemonic::from_str(
            "trophy network surprise adjust flame glance swift impose urge way ozone nose agent copy feed away spare slide soul drama dinosaur project issue repeat",
        )
        .expect("Failed to generate mnemonic");
        let index = 489404;

        let expected_address = "0x0000007D77bB7c2D4eE6d077ee463f8c340Ea268";
        let address = eth_address_from_mnemonic(&mnemonic, index);

        println!("Mnemonic: {}", mnemonic);
        println!("Child Index: {}", index);
        println!("Ethereum Address: {}", address);
        assert_eq!(address, expected_address);
    }
}
