import mne

mne.set_log_level('ERROR')


def exclude_EOG_ECG_with_ICA(raw, semiauto=False):
    """
    Apply ICA to remove EOG and ECG artifacts from raw instance.

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
    eog_scores : Scores used for selection of the EOG component(s).
    ecg_scores : Scores used for selection of the ECG component(s).
    """
    # ICA decomposition
    ica = mne.preprocessing.ICA(method='picard', max_iter='auto')
    ica.fit(raw, picks='eeg')

    # EOG
    eog_idx, eog_scores = ica.find_bads_eog(raw, threshold=0.6,
                                            measure='correlation')
    # ECG
    ecg_idx, ecg_scores = ica.find_bads_ecg(raw, method='correlation',
                                            threshold=6.6, measure='zscore')

    # Exclude occular and heartbeat related components
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
    return raw, ica, eog_scores[eog_idx], ecg_scores[ecg_idx]
