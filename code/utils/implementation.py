"""
Scelta dell'implementazione della similarità (Task 2 di workflow/specs/scale-test.md).

Il valore di default riproduce esattamente il codice usato nelle campagne. L'alternativa si attiva solo dove serve
(prova di scala) chiamando configure() all'avvio del processo, prima di creare gli environment. Non è un campo delle
impostazioni delle istanze: cambiarla non cambia gli id dei run.

- FAST_SIMILARITY = True: la similarità si calcola solo sulle coppie che condividono almeno una risorsa (indice
  inverso "risorsa -> entità"), con la stessa funzione di oggi; le altre coppie valgono 0 come prima. Risultato
  identico bit per bit (verificato con le impronte della sonda il 28/11).

NJ resta quello originale (clustering/nj/nj_torch.py): la versione con meno copie era più lenta sulla GPU e la
precisione singola cambiava alcune partizioni per circa il 28% di tempo in meno; tolte il 28/11 (H22).
"""

FAST_SIMILARITY = False


def configure(fast_similarity: bool | None = None):
    global FAST_SIMILARITY
    if fast_similarity is not None:
        FAST_SIMILARITY = bool(fast_similarity)


def describe() -> dict:
    """Implementazione in uso, da registrare insieme ai risultati."""
    return {"impl_fast_similarity": FAST_SIMILARITY}
