/** Base64 <-> binary helpers. Pure JS — Hermes-safe, no atob/Buffer dependency. */

const B64_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

const B64_LOOKUP = (() => {
  const table = new Int8Array(128).fill(-1);
  for (let i = 0; i < B64_CHARS.length; i++) table[B64_CHARS.charCodeAt(i)] = i;
  return table;
})();

export function base64ToBytes(b64: string): Uint8Array {
  let len = b64.length;
  while (len > 0 && (b64[len - 1] === '=' || b64[len - 1] === '\n')) len--;
  const outLen = Math.floor((len * 3) / 4);
  const out = new Uint8Array(outLen);

  let o = 0;
  let buffer = 0;
  let bits = 0;
  for (let i = 0; i < len; i++) {
    const c = B64_LOOKUP[b64.charCodeAt(i)];
    if (c < 0) continue; // skip whitespace / invalid
    buffer = (buffer << 6) | c;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out[o++] = (buffer >> bits) & 0xff;
    }
  }
  return o === outLen ? out : out.subarray(0, o);
}

/** Interprets bytes as little-endian 16-bit PCM samples. */
export function bytesToInt16(bytes: Uint8Array): Int16Array {
  // Align: Int16Array view needs an even byte offset/length
  const even = bytes.byteLength & ~1;
  if (bytes.byteOffset % 2 === 0) {
    return new Int16Array(bytes.buffer, bytes.byteOffset, even / 2);
  }
  const copy = bytes.slice(0, even);
  return new Int16Array(copy.buffer, 0, even / 2);
}
