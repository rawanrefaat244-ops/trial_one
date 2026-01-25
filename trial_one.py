from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def parse_pssm(pssm_file):
    with open(pssm_file) as f:
        lines = f.readlines()

    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 22:
            continue
        try:
            scores = [int(val) for val in parts[2:22]]
            data.append(scores)
        except ValueError:
            continue

    return np.array(data)

if __name__ == "__main__":

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

    x = parse_pssm("3ave.pssm")

    label_map = {'H': 0, 'E': 1, 'C': 2}
    y = np.array([label_map[ss] for ss in secondary_structure])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    y_pred_full = model.predict(x)

    pred_labels_full = ['H' if label == 0 else 'E' if label == 1 else 'C' for label in y_pred_full]
    predicted_secondary_structure_full = ''.join(pred_labels_full)

    print("\nPredicted Secondary Structure (H/E/C):")
    print(predicted_secondary_structure_full)

    print('\nCounts:')
    print('Helix (H):', predicted_secondary_structure_full.count('H'))
    print('Strand (E):', predicted_secondary_structure_full.count('E'))
    print('Coil (C):', predicted_secondary_structure_full.count('C'))

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
