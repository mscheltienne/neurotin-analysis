import mne

mne.set_log_level('ERROR')


def _ica(raw, **kwargs):
    """Fit an ICA with the given kwargs to raw EEG channels.
    kwargs are passed to mne.preprocessing.ICA()."""
    ica = mne.preprocessing.ICA(method='picard', max_iter='auto')
    ica.fit(raw, picks='eeg')
    return ica


def _exclude_ocular_components(raw, ica, **kwargs):
    """Find and exclude ocular-related components.
    kwargs are passed to ica.find_bads_eog()."""
    eog_idx, eog_scores = ica.find_bads_eog(raw, **kwargs)
    return eog_idx, eog_scores[eog_idx]


def _exclude_heartbeat_components(raw, ica, **kwargs):
    """Find and exclude heartbeat-related components.
    kwargs are passed to ica.find_bads_ecg()."""
    ecg_idx, ecg_scores = ica.find_bads_ecg(raw, **kwargs)
    return ecg_idx, ecg_scores[ecg_idx]


def exclude_ocular_and_heartbeat_with_ICA(raw, semiauto=False):
    """
    Apply ICA to remove ocular and heartbeat artifacts from raw instance.

    Parameters
    ----------
    raw : raw : Raw
        Raw instance modified in-place.
    semiauto : bool
        If True, the user will interactively exclude ICA components if
        automatic selection failed.

    Returns
    -------
    raw : Raw instance modified in-place.
    ica : ICA instance.
    eog_scores : Scores used for selection of the ocular component(s).
    ecg_scores : Scores used for selection of the heartbeat component(s).
    """
    ica = _ica(method='picard', max_iter='auto')

    eog_idx, eog_scores = \
        _exclude_ocular_components(raw, ica, threshold=0.6,
                                   measure='correlation')
    ecg_idx, ecg_scores = \
        _exclude_heartbeat_components(raw, ica, method='correlation',
                                      threshold=6.6, measure='zscore')

    ica.exclude = eog_idx + ecg_idx

    try:
        assert len(eog_idx) <= 2, 'More than 2 EOG component detected.'
        assert len(ecg_idx) <= 1, 'More than 1 ECG component detected.'
        assert len(ica.exclude) != 0, 'No EOG/ECG component detected.'
    except AssertionError:
        if semiauto:
            ica.plot_scores(eog_scores)
            ica.plot_scores(ecg_scores)
            ica.plot_sources(raw, block=True)
        else:
            raise

    # Apply ICA
    ica.apply(raw)

    # Scores should not be returned when #9846 is fixed.
    return raw, ica, eog_scores, ecg_scores
