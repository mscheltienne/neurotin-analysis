from matplotlib import pyplot as plt

from neurotin.time_frequency.viz import plot_tfr_subject


folder_tfr = "/Users/scheltie/Documents/datasets/neurotin/tfr/"


#%% Full
# 65
plot_tfr_subject(
    folder_tfr,
    65,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(11, 9), (12, 10), (14, 9)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/065.svg')

# 66
plot_tfr_subject(
    folder_tfr,
    66,
    "multitaper",
    resolutions=(2, 2),
    timefreqs=[(17, 6), (22, 6), (18, 12)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/066.svg')

# 68
plot_tfr_subject(
    folder_tfr,
    68,
    "multitaper",
    resolutions=(2, 2),
    timefreqs=[(10, 10), (14, 10)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/068.svg')

# 72
plot_tfr_subject(
    folder_tfr,
    72,
    "multitaper",
    resolutions=(2, 2),
    timefreqs=[(12, 13), (16, 13)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/072.svg')

# 73
plot_tfr_subject(
    folder_tfr,
    73,
    "multitaper",
    resolutions=(2, 2),
    timefreqs=[(14, 12), (16, 12), (18, 12)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/073.svg')

# 75
plot_tfr_subject(
    folder_tfr,
    75,
    "multitaper",
    resolutions=(2, 2),
    timefreqs=[(12, 10), (14, 10)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/075.svg')


#%% Transfer
plot_tfr_subject(
    folder_tfr,
    57,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(16, 11), (14, 1), (16, 2), (21, 2)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/057-transfer.svg')

plot_tfr_subject(
    folder_tfr,
    61,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(12, 8), (16, 2), (21, 2)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/061-transfer.svg')

plot_tfr_subject(
    folder_tfr,
    63,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(11, 10), (16, 2), (15, 10), (21, 2)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/063-transfer.svg')

plot_tfr_subject(
    folder_tfr,
    65,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(11, 9), (12, 10), (16, 2), (21, 2)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/065-transfer.svg')

plot_tfr_subject(
    folder_tfr,
    66,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(17, 6), (18, 12), (22, 6), (22, 12)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/066-transfer.svg')

plot_tfr_subject(
    folder_tfr,
    68,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(10, 10), (15, 10), (20, 10), (13, 2)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/068-transfer.svg')

plot_tfr_subject(
    folder_tfr,
    68,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(13, 2), (20, 2)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/068-transfer-lowfq.svg')

plot_tfr_subject(
    folder_tfr,
    69,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(10, 10), (15, 10), (20, 10), (13, 10)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/069-transfer.svg')

plot_tfr_subject(
    folder_tfr,
    72,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(10, 12), (15, 12), (20, 12), (13, 12)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/072-transfer.svg')

plot_tfr_subject(
    folder_tfr,
    73,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(15, 12), (20, 12), (13, 12), (22, 12)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/073-transfer.svg')

plot_tfr_subject(
    folder_tfr,
    75,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(15, 10), (20, 10), (13, 10), (22, 10), (12, 1), (20, 1)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/075-transfer.svg')

plot_tfr_subject(
    folder_tfr,
    76,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(15, 12), (20, 12), (13, 2), (22, 2)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/076-transfer.svg')

plot_tfr_subject(
    folder_tfr,
    78,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(15, 8), (12, 8), (13, 2), (22, 2)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/078-transfer.svg')

plot_tfr_subject(
    folder_tfr,
    79,
    "multitaper",
    resolutions=(2, 2),
    transfer_only=True,
    timefreqs=[(15, 9), (12, 9), (13, 2), (22, 2)],
)
plt.savefig('/Users/scheltie/Documents/datasets/neurotin/viz/tfrs/079-transfer.svg')
