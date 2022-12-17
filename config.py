import configparser

models = ['pls', 'lasso', 'pcr']
preprocessors = ['savgol0', 'savgol1', 'savgol2', 'snv', 'msc', 'savgol0+snv', 'savgol1+snv', 'savgol2+snv',
                 'savgol0+msc', 'savgol1+msc', 'savgol2+msc']
evaluator = 'mse'

config = configparser.ConfigParser()
counter = 0

for m in models:
    for p in preprocessors:
        counter += 1
        # Add the structure to the file we will create
        config.add_section('pipeline'+str(counter))
        config.set('pipeline'+str(counter), 'preprocessor', p)
        config.set('pipeline'+str(counter), 'model', m)
        config.set('pipeline'+str(counter), 'evaluator', evaluator)

# Write the new structure to the new file
with open(r"configfile.ini", 'w') as configfile:
    config.write(configfile)