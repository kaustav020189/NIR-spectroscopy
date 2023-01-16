# NIR-spectroscopy using Python
##### Project title - Regression Problems with Photonic Data in Automation

## Dependencies
##### Python 3.+ , pip / conda

## Initial setup
Keep all the files from this repository in the same directory. Keep the 4 data files (.csv) in this directory as well. All model pipelines can be 
run from the two .ipynb files itself. No other file access is required.

## Understanding the code
- The two .ipynb files contain pipelines for PLS and Lasso respectively.
- Each pipeline code block has an associated markup block before it to describe the pipeline features
- The code blocks have complete comments section for each code statements. Follow the comments if required.
- The util.py contains all custom function code and is the background for all the pipeline runs. It is not necessary to access this code, since every
variation of execution can be handled from the .ipynb itself.

## Running the code
- Start a Jupyter notebook environment
- Running all the code blocks at once can take a huge time, instead read the description of each pipeline block and run one at a time
- **Always run the first two code blocks in a .ipynb file before running any pipeline code. Otherwise it might throw an error**
The first two blocks import and initiate all necessary variables and modules.
- Each pipeline has parameters already set as per the description of the pipeline (given in the markup above). Changing that is possible at all times, 
but must be done following the data-type check, etc. 
- Most pipeline parameters are values retrieved from predefined variables in the configfile.ini
- Check the configfile.ini for the various different options available when setting parameter values.

### Parameter help for the util.pipeline() method
For modifying and getting variations in the call, only changes to the below keys are required in the __pipeline_params__ variable

**Refer the configfile.ini file to understand which index you should use in order to supply a particular value**

- `validation-type` changes validation method. e.g. - validation_types[0] for KFolds CV
- `preprocessor` changes pre-processing method to be used. e.g. - preprocessor[3] for SNV
Some pipelines have multiple pre-processors implemented using for-loops. You can modify the for-loop list as per your choice to select some particular ones.
- `model` changes the regression algorithm. e.g - model[0] for PLS and model[1] for Lasso
- `plotting` - set True to show plots in all stages
- `showModelEvaluationPlots` - set True to show final best fit line plot
- `pipeline_run_type` - sets whether full pipeline is executed or only the preprocessing part. e.g. - pipeline_run_type[0] means only preprocessing

### Identifying which parts to change when making a pipeline call
Only changes to the following sections are required in order to modify pipeline calls

- Make changes to `dataset = files[3]` if changing the no. of files for the pipeline (refer comments above this line)
- Make changes to SavGol + derivative call by changing the derivative (0,1 or 2). Change the value for the key 'derivative' 

**Suggested : Not to change the other two key:value pairs**
```
preprocessor_params = {
        'savgol' : {
            'window_size_range' : np.arange(3,63,3),
            'polyorder_range' : np.arange(2,18,2),
            'derivative' : 1
        }
    }
```
- The main pipeline_params variable, where most of the modfications can be done. Refer previous section "Parameter help" to understand what to change.
```
pipeline_params = {
        'logger' : logger,
        'data' : data,
        'validation-type' : validation_types[0],
        'preprocessor' : preprocessor[3],
        'preprocessor-params' : preprocessor_params,
        'model' : models[0],
        'evaluator' : evaluators[0],
        'pls_max_n_comp' : int(pls_max_n_comp),
        'folds' : int(folds),
        'wl' : wl,
        'plotting' : False,
        'showModelEvaluationPlots' : True,
        'pipeline_run_type' : pipeline_run_types[0]
    }
```
