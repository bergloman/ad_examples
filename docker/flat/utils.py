from tdigest import TDigest

def calculateQuantiles(col, min_tdigest_items):
    digest = TDigest()

    for i in range(min_tdigest_items):
     digest.update(col[i])

    quantiles = []
    for i in range(len(col)):
        val = col[i]
        c = digest.cdf(val)
        quantiles.append(c)
        if i >= min_tdigest_items:
            digest.update(val)

    return quantiles


def calculateQuantilesForDf(df, min_tdigest_items, cols_to_skip):
    cols = list(df.columns)
    cols_to_process = [x for x in cols if (x in cols_to_skip) == False]
    for x in cols_to_process:
        df[x + "_q"] = calculateQuantiles(df[x], min_tdigest_items)
