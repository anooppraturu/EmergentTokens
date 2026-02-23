# Collection of functions to streamline getting raw word representations from residual stream of transformer

import torch
import transformer
import numpy as np

def truncate(tok_dat):
    """
    in: tokenized input tensor (ctx_len)
    out: tokenized tensor (T<=ctx_len) with first and last word removed (so there are no partial words truncated by context window)
    """
    # 0 is whitespace token
    space_locs = torch.where(tok_dat == 0)[0]
    truncated = tok_dat[space_locs[0]+1:space_locs[-1]]
    return truncated[None, :]


def reconstruct_string(toks, id_to_char):
    """
    convert tensor of tokens to string of text
    
    :param toks: 1D tensor of tokens
    :param id_to_char: dictionary mapping tokens to characters
    """
    return ('').join([id_to_char[idx.item()] for idx in toks])


def get_input_occurences(toks, top_words, word_ids, id_to_char):
    """
    in: truncated 1D tensor of model input tokens
        list of words foe classifier dataset
        dictionary mapping words to ids
    out: occurences list [o1, o2, o3, ..., oN]
        where oi = (start_id, end_id, word_id) is tuple of start and end index of word with id word_id
    """
    string = reconstruct_string(toks, id_to_char)
    wds = string.split(' ')
    occurences = []

    for i, wd in enumerate(wds):
        if wd in top_words:
            recons = (' ').join(wds[:i])
            start_id = len(recons) + 1
            end_id = start_id + len(wd)
            occurences.append((start_id, end_id, word_ids[wd]))

    return occurences


def pad_residual(res_dat, occ_dat, full_len=7):
    """
    in: full (T, d_res) (where T <= ctx length) residual tensor at fixed layer
        (start_id, end_id, word_id) occurence data
        length word tensor should be padded to
    out: (full_len, d_res) tensor of character residuals in words, where first entries are masked to 0
    """
    _, d_res = res_dat.shape
    start, end, _ = occ_dat
    word_len = end - start

    word_residuals = res_dat[start:end]
    N_rem = full_len - word_len
    assert(N_rem >= 0)
    if N_rem == 0:
        return word_residuals
    
    pad = torch.zeros(N_rem, d_res)
    padded = torch.concat((pad, word_residuals), dim=0)

    return padded


def get_all_occurences(raw_tokens, model, top_words, word_ids, id_to_char):
    """
    in: raw input list of tokens from dataset
        model to evaluate on
    out: layer len list of datasets [ds1, ds2, ..., dsL]
        where ds1 = [(X1, y1), (X2, y2), ..., (XN, yN)]
        Xi is (full_len=7, d_res) tensor of residual representation of word yi
        at the specified layer
    """
    model.enable_residual_logging()

    truncated_tokens = truncate(raw_tokens)
    occurences = get_input_occurences(truncated_tokens[0], top_words, word_ids, id_to_char)

    #run forward pass to get residuals
    with torch.no_grad():
        _ = model(truncated_tokens)

    layer_datasets = []
    for res in model.residuals[1:]:
        ds = []
        for occ in occurences:
            X = pad_residual(res[0], occ)
            y = occ[2]
            ds.append((X, y))

        layer_datasets.append(ds)
    
    model.disable_residual_logging()

    return layer_datasets
