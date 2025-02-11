import gmtkn55.sets as sets

dict = sets.IDISP.systems
def input_set(sets):
    global dict
    dict = sets

def getAtom(atom):
    atm = ''
    length = len(dict[atom]['atoms'])
    for i in range(length):
        atm += dict[atom]['atoms'][i]
        atm += ' '
        for j in range(len(dict[atom]['coords'][i])):
            atm += str(dict[atom]['coords'][i][j])
            atm += ' '
        if i != length-1:
            atm += ';'

    charge = dict[atom]['charge']
    spin = dict[atom]['spin']

    return atm, spin, charge

