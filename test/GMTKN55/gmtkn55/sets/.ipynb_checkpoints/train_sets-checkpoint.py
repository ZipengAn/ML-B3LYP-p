systems = {
    'h' : { 'atoms' : ['H'], 'coords' : [[0.0, 0.0, 0.0]], 'charge' : 0, 'spin' : 1 },
    'c' : { 'atoms' : ['C'], 'coords' : [[0.0, 0.0, 0.0]], 'charge' : 0, 'spin' : 2 },
    'o' : { 'atoms' : ['O'], 'coords' : [[0.0, 0.0, 0.0]], 'charge' : 0, 'spin' : 2 },
    's' : { 'atoms' : ['S'], 'coords' : [[0.0, 0.0, 0.0]], 'charge' : 0, 'spin' : 2 },
    'nh3' : { 'atoms' : ['N', 'H', 'H', 'H'], 'coords' : [[0.0, 0.0, 0.116671], [0.0, 0.934724, -0.272232], [0.809495, -0.467362, -0.272232], [-0.809495, -0.467362, -0.272232]], 'charge' : 0, 'spin' : 0 },
    'h2o' : { 'atoms' : ['O', 'H', 'H'], 'coords' : [[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047], [0.0, -0.763239, -0.477047]], 'charge' : 0, 'spin' : 0 },
    'so2' : { 'atoms' : ['S', 'O', 'O'], 'coords' : [[0.0, 0.0, 0.370268], [0.0, 1.277617, -0.370268], [0.0, -1.277617, -0.370268]], 'charge' : 0, 'spin' : 0 },
    'c2h2' : { 'atoms' : ['H', 'C', 'C', 'H'], 'coords' : [[0.0, 0.0, 1.673990], [0.0, 0.0, 0.608080], [0.0, 0.0, -0.608080], [0.0, 0.0, -1.673990]], 'charge' : 0, 'spin' : 0 },
    'co2' : { 'atoms' : ['C', 'O', 'O'], 'coords' : [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1626], [0.0, 0.0, -1.1626]], 'charge' : 0, 'spin' : 0 }
}

reactions = [
    { 'systems' : ['h2o', 'h', 'o'], 'stoichiometry' : ['-1', '2', '1'], 'reference' : 232.974000 },
    { 'systems' : ['so2', 's', 'o'], 'stoichiometry' : ['-1', '1', '2'], 'reference' : 260.621000 },
    { 'systems' : ['c2h2', 'c', 'h'], 'stoichiometry' : ['-1', '2', '2'], 'reference' : 405.525000 },
    { 'systems' : ['nh3', 'n', 'h'], 'stoichiometry' : ['-1', '1', '3'], 'reference' : -35491.8 },
    { 'systems' : ['co2', 'c', 'o'], 'stoichiometry' : ['-1', '1', '2'], 'reference' : 390.141000 },
    { 'systems' : ['co2'], 'stoichiometry' : ['1'], 'reference' : -118341.3 },
    { 'systems' : ['h2o'], 'stoichiometry' : ['1'], 'reference' : -47963.0 },
    { 'systems' : ['nh3'], 'stoichiometry' : ['1'], 'reference' : -47963.0 },
    { 'systems' : ['so2'], 'stoichiometry' : ['1'], 'reference' : -48524.5 },
    { 'systems' : ['h2o'], 'stoichiometry' : ['1'], 'reference' : -47963.0 },
    { 'systems' : ['h'], 'stoichiometry' : ['1'], 'reference' : -344244.0 },
    { 'systems' : ['c'], 'stoichiometry' : ['1'], 'reference' : -313.8 },
    { 'systems' : ['n'], 'stoichiometry' : ['1'], 'reference' : -23745.8 },
    { 'systems' : ['o'], 'stoichiometry' : ['1'], 'reference' : -47102.5 },
    { 'systems' : ['s'], 'stoichiometry' : ['1'], 'reference' : -249778.1 },

]

