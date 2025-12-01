from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

parser = PDBParser(QUIET=True)
structure = parser.get_structure('3AVE', '3ave.pdb')
model = structure[0]


dssp = DSSP(model, '3ave.pdb', dssp='mkdssp')

sequence = []
ss_labels = []

for key in dssp.keys():
    aa = dssp[key][1]
    ss = dssp[key][2]
    if ss in ['H', 'G', 'I']:
        ss = 'H'
    elif ss in ['E', 'B']:
        ss = 'E'
    else:
        ss = 'C'
    sequence.append(aa)
    ss_labels.append(ss)

primary_sequence = ''.join(sequence)
secondary_structure = ''.join(ss_labels)

print('\nPrimary sequence:')
print(primary_sequence)

print('\nExperimental secondary structure (H/E/C):')
print(secondary_structure)

print('\nCounts:')
print('Helix (H):', secondary_structure.count('H'))
print('Strand (E):', secondary_structure.count('E'))
print('Coil (C):', secondary_structure.count('C'))

with open("3ave.fasta", "w") as fasta_file:
    fasta_file.write(">3AVE\n")
    fasta_file.write(primary_sequence + "\n")

# x = parse_pssm("3ave.pssm")
#
# label_map = {'H': 0, 'E': 1, 'C': 2}
# y = np.array([label_map[ss] for ss in secondary_structure])
#
# print(f"PSSM shape: {x.shape}")
# print(f"Labels shape: {y.shape}")
#
# def create_windows(x, y, window_size=17):
#     half = window_size // 2
#     padded_x = np.pad(x, ((half, half), (0, 0)), 'constant', constant_values=0)
#
#     x_windowed = []
#     y_windowed = []
#
#     for i in range(len(y)):
#         window = padded_x[i:i+window_size].flatten()
#         x_windowed.append(window)
#         y_windowed.append(y[i])
#
#     return np.array(x_windowed), np.array(y_windowed)
#
# x_win, y_win = create_windows(x, y, window_size=17)
# print(f"Windowed input shape: {x_win.shape}")
# print(f"Windowed labels shape: {y_win.shape}")



