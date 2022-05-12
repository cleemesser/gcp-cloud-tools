from utils.config import flag, chain, prod, lzip

def s4_embedding_repro():
    return prod(
        [
            flag("experiment", ['s4-qautomusic']),
            prod(
                [
                    flag("dataset.quantization", ['linear', 'mu-law']),
                ]
            ),
        ]
    )
