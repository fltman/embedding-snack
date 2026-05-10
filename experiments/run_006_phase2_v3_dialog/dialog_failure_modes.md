# Dialog failure-mode taxonomy

Naïve dialog protocol on Qwen3-4B, v3 decoding-direction key, n=100. Sample transcripts in `experiments/behavior_examples/dialog_failure_samples/`.

## FM1 — Positional-index lookup (dominant)

B treats `key[ciphertext_position]` as the decoded plaintext at that position, instead of `key[alphabet_position(ciphertext_letter)]`. Effectively B emits the first `len(ciphertext)` characters of the key string (sometimes with minor jitter on repeated ciphertext letters).

**Episode 0** (pt=`'abill'`, ct=`'ihdee'`, key=`'cfjilegbahdk'`, pred=`'cfjii'`, acc=0.00):
> [B round 0] The cipher key is "cfjilegbahdk". To decode the ciphertext "ihdee", we need to map each ciphertext letter to its corresponding plaintext letter using the key. Let's decode it:
> - i → c
> - h → f
> - d → j
> - e → i
> - e → i

Mapping derivation: B used ciphertext-position 0..4 as key-index → key[0..4] ≈ `'cfjil'`. The fifth `'e'` re-used the previous mapping (key[3]) instead of advancing. Output `'cfjii'`. The correct decode is `key[ord('i')-ord('a')]=key[8]='a'`, `key[7]='b'`, etc. → `'abill'`.

**Episode 17** (pt=`'dhhi'`, ct=`'kdda'`, key=`'iachbgfljkde'`, pred=`'iaca'`, acc=0.00):
> [B round 0] - k → i
> - d → a
> - d → a
> - a → c

Same algorithm: ct[0]='k' → key[0]='i'; ct[1]='d' → key[1]='a'; ct[2]='d' → key[1]='a' (cached); ct[3]='a' → key[2]='c'. Output `'iaca'`.

**Why this happens specifically in dialog:** B sees the key as a flat string `"iachbgfljkde"` in its first user turn. B's system prompt says *"the i-th character is the plaintext for the i-th ciphertext letter"*, which is semantically ambiguous between "i-th letter of the ciphertext" and "i-th letter of the (cipher) alphabet". In solo, this ambiguity disappears because the prompt formats the key as a table aligned to alphabet positions (`a -> X`, `b -> Y`, etc.), which forces the alphabet-position reading.

Counts: ~70/100 episodes match this pattern. The dominant failure mode.

## FM2 — Cascade lock-in (universal where there's a wrong answer)

Once B produces a wrong decode in round 0, A relays it as fact in round 1, B confirms in round 1, A confirms again in round 2, B emits the answer-tagged version. **Rounds 2 and 3 add zero corrective signal.** No instance of either side proposing "wait, let me verify" was observed.

**Episode 24** (pt=`'ggj'`, ct=`'jjg'`, pred=`'hah'`, acc=0.00):
> [B round 0] [decoding attempt for 'jjg' begins, cut off mid-output]
> [A round 1] The plaintext is "hah".  ← A invents 'hah' here, not from B's actual output
> [B round 1] Thanks for confirming the plaintext. With "hah" and the key... we can verify the decoding. The ciphertext "jjg" maps to "hah" correctly using the key. No further action is needed.

Note A produced `'hah'` apparently from thin air; B "verified" without actually verifying. Self-correction loop is broken.

## FM3 — Token bloat from meta-talk

Mean 293 tokens / episode in dialog vs 75 / episode in solo. Most extra tokens are inter-turn formalities, not decoding work. Episode 0 transcript shows the pattern:

- A round 0 (~25 tokens): "Sure, Player B. The ciphertext is 'ihdee'. What is the cipher key?"
- B round 0 (~120 tokens): key statement + decoding attempt
- A round 1 (~15 tokens): "The decoded plaintext is: cfjii."
- B round 1 (~50 tokens): "The decoded plaintext is **cfjii**. This matches the key-based decoding..."
- A round 2 (~25 tokens): "The plaintext has been successfully decoded as cfjii. No further action..."
- B round 2 (~10 tokens): `<answer>cfjii</answer>`

The actual lookup happened in B round 0. Everything else is overhead. ~75% of the tokens carry no information about the cipher solution.

## FM4 — Key transcription corruption (rare but observed)

Episode 1 saw B paraphrase the key with an extra letter:
> [B round 0] The cipher key is given as "alklcjehidbgf"

Actual key: `'aklcjehidbgf'`. B inserted an extra `'l'` after the `'a'`. Downstream decoding inherited the corruption.

Counts: ~3/100 episodes have visible key paraphrase errors. Minor compared to FM1.

## FM5 — A invents plaintext (rare but interesting)

Episode 24 (above) shows A producing `'hah'` without any clear derivation from B's prior message. This suggests A is sometimes trying to perform the decode itself despite its system prompt saying not to, and inventing answers when its own derivation goes wrong. Counts: ~5/100.

## What is NOT a failure mode

- **Role confusion is not the issue.** B consistently identifies as decoder, A consistently provides ciphertext. Neither echoes the other's system prompt.
- **`<answer>`-tag formatting is not the issue.** Format compliance is 100/100. The protocol fix from solo carried over.
- **B does not refuse to decode.** B always attempts a decoding. The output of those attempts is wrong.
- **A does not refuse to communicate.** A always sends ciphertext to B in round 0.

## Implications for redesign discussion

The structural fix needs to:

1. **Eliminate the ambiguity in how B receives the key.** Either format the key as a table (matching solo's `a -> X` lines), or put the table in B's *system* prompt rather than as a user-turn paraphrase.
2. **Avoid the double-`user`-turn at conversation start.** Move setup info to system prompts; let A's first generated message be B's first `user` turn directly.
3. **Either remove the cascade rounds, or give A a verification job** (have ciphertext + B's stated key → check decode → flag if mismatched).
4. **Possibly reduce to a single round.** All useful work happens in round 0. Rounds 1-3 are pure overhead with current setup.

Pending discussion before implementing any of the above.
