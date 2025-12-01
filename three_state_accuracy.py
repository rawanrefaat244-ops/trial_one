exp = "CCCCCEEEEECCCHHHHHCCCCCCEEEEEEEEEECCCCCEEEEEEECCEEECCCEECCCEECCCCEEEEEEEEECCHHHHHCCCCEEEEEEECCCCCCEEEEECCCCCCCECCEEEEECCCHHHHHCCEEEEEEEEEEEECCCCEEEEEECCEECCCEEECCCEECCCCCEEEEEEEEEEHHHHHCCCCEEEEEECCCCHHHEEEEEECCCCCEEEEECCCHHHHHCCCCCCEEEEEEEEECCCCCCCEEEEEECCEEECCCEECCCEECCCCCEEEEEEEECCHHHHHCCCCEEEEEECCCCCCCEEEEECCCCCCCECCEEEEECCCHHHHHCCEEEEEEEEEEEECCCCEEEEEECCEECCCEEECCCEECCCCCEEEEEEEEEEHHHHHCCCCEEEEEECCCCHHHEEEEEECCC"
exp = exp.replace('C', '-')

pred = "------EEEE-----EEEEE-----EEEEEEEE------EEEEEEE---E-------EEE------EEEEEEEEEEE-------EEEEEEEE-------EEEEEE--------EEEEE-----------EEEEEEEE------EEEEEE------------EEEE-------EEEEEEEE--------EEEEEEEE------EEEEEEEE-------------------------------------------------------------------------EEE----------EEEE----------EEE-------------------------EEEEEE--------EEEEE----EE------EEEE----EEEEEEEEEE---------EEEEEEEE------EEEEEEE--"

min_len = min(len(exp), len(pred))
exp = exp[:min_len]
pred = pred[:min_len]

correct = sum(1 for e, p in zip(exp, pred) if e == p)
accuracy = (correct / min_len) * 100

print(f'Prediction Match: {correct}/{min_len} residues')
print(f'Three-state accuracy: {accuracy:.2f}%')

print(len(exp) == len(pred))

