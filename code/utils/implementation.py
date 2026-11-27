"""
Scelta dell'implementazione di similarità e NJ (Task 2 di workflow/specs/scale-test.md).

I valori di default riproducono esattamente il codice usato nelle campagne. Le alternative si attivano solo dove
servono (prova di scala) chiamando configure() all'avvio del processo, prima di creare gli environment. Non sono
campi delle impostazioni delle istanze: cambiarle non cambia gli id dei run.

- FAST_SIMILARITY = True: la similarità si calcola solo sulle coppie che condividono almeno una risorsa (indice
  inverso "risorsa -> entità"), con la stessa funzione di oggi; le altre coppie valgono 0 come prima. Risultato
  identico bit per bit.
- NJ_IMPLEMENTATION = "lean": NJ con le stesse operazioni aritmetiche nello stesso ordine dell'originale, ma con meno
  copie e allocazioni di matrici n x n a ogni passo. Risultato identico (stesso albero, stessi pareggi).
- NJ_DTYPE = "float32": NJ in precisione singola (metà memoria e metà traffico). NON identico: gli arrotondamenti
  cambiano e la partizione può cambiare; da usare solo dopo averne misurato l'effetto.
"""

FAST_SIMILARITY = False
NJ_IMPLEMENTATION = "original"   # "original" o "lean"
NJ_DTYPE = "float64"             # "float64" (identico) o "float32" (non identico)

_VALID_NJ = ("original", "lean")
_VALID_DTYPE = ("float64", "float32")


def configure(fast_similarity: bool | None = None, nj_implementation: str | None = None, nj_dtype: str | None = None):
    global FAST_SIMILARITY, NJ_IMPLEMENTATION, NJ_DTYPE
    if fast_similarity is not None:
        FAST_SIMILARITY = bool(fast_similarity)
    if nj_implementation is not None:
        if nj_implementation not in _VALID_NJ:
            raise ValueError(f"nj_implementation '{nj_implementation}' non valida: {_VALID_NJ}")
        NJ_IMPLEMENTATION = nj_implementation
    if nj_dtype is not None:
        if nj_dtype not in _VALID_DTYPE:
            raise ValueError(f"nj_dtype '{nj_dtype}' non valido: {_VALID_DTYPE}")
        NJ_DTYPE = nj_dtype


def nj_implementation_in_use() -> str:
    """NJ davvero eseguito: la precisione singola esiste solo nella versione lean (vedi NjTree.__init__)."""
    return "lean" if NJ_IMPLEMENTATION == "lean" or NJ_DTYPE != "float64" else "original"


def describe() -> dict:
    """Implementazione in uso, da registrare insieme ai risultati."""
    return {"impl_fast_similarity": FAST_SIMILARITY, "impl_nj": nj_implementation_in_use(), "impl_nj_dtype": NJ_DTYPE}
