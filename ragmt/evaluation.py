from sacrebleu.metrics import BLEU, CHRF, TER


def mt_evaluate(decoded_preds, decoded_labels):
    bleu = (
        BLEU(tokenize="flores200").corpus_score(decoded_preds, [decoded_labels]).score
    )
    chrf = CHRF().corpus_score(decoded_preds, [decoded_labels]).score
    ter = TER().corpus_score(decoded_preds, [decoded_labels]).score
    metrics = {
        "bleu": bleu,
        "chrf": chrf,
        "ter": ter,
    }

    return metrics
