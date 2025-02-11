# ML-B3LYP-p Program README

This repository contains the code for the ML-B3LYP-p program and model described in the associated paper.  
**Dependencies**:  
- **PySCF** (version `2.0.1` recommended)  
- **PyTorch** (version `1.13.0+cpu` or higher)  

Install via `pip`:  
# Install PySCF
pip install pyscf==2.0.1

# Install PyTorch (CPU version, adjust the command for your OS if needed)
pip install torch==1.13.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu


# Program Structure and usage:

The program is organized into three main functional modules as follows:

---

## 1. `train` Folder: Model Training
This folder contains the code used to train and generate the optimized ML-B3LYP-p model (note: the program and model share the same name; refer to the paper for clarification).

### Usage:
1. **Path Configuration**:  
   Modify the `basis_path` variable in `train_bp.py` to your local code directory path (replace `/your/path/to/put/this/code`).

2. **Execution**:  
   Run the training script with a specified learning rate:  
   ```
   python train_bp.py [learning_rate]
   ```  
   Example:  
   ```
   python train_bp.py 0.001
   ```

3. **Output**:  
   - Multiple `.log` files will be generated, containing model parameters saved at each convergence step.  
   - An `info_xxx.log` file (where `xxx` is the learning rate) records training/validation metrics.  
   - To identify the best model before overfitting, use:  
     ```
     grep "Best:.*_SCF" info_*.log
     ```

---

## 2. `test/TEST12` Folder: TEST12 Benchmark Testing
Code for evaluating the model on the TEST12 dataset. Note: Input formats differ between TEST12 and GMTKN55, necessitating separate test scripts.

### Usage:
1. **Path Configuration**:  
   Update the `basis_path` variable in `test.py` to your local code directory path (replace `/your/path/to/put/this/code`).

2. **Execution**:  
   Run the test script with a target subset and the `b3lyp` flag:  
   ```
   python test.py [subset_name] b3lyp
   ```  
   Example (for the `AE6` subset):  
   ```
   python test.py AE6 b3lyp
   ```

3. **Output**:  
   - `ae_*.log`: Relative energies of molecules/atoms in the subset.  
   - `te_*.log`: Absolute energies of molecules/atoms in the subset.  
   - `info_*.log`: Detailed testing metrics.  

4. **Post-Processing**:  
   - Copy the second column of relative energies from `ae_*.log` into the **"Origin data"** sheet of the provided `test.xlsx`.  
   - For `G2-EA` and `G2-IP` subsets: Compute `G2-add` and populate the corresponding column under **"G2-AE"** in the Excel sheet.

---

## 3. `test/GMTKN55` Folder: GMTKN55 Benchmark Testing
Code for evaluating the model on the GMTKN55 dataset.

### Usage:
1. **Path Configuration**:  
   Modify the `output` path in `test.py` to your local code directory (replace `/your/path/to/put/this/code`).

2. **Execution**:  
   Run the script without additional arguments:  
   ```
   python test.py
   ```

3. **Output**:  
   Results for all subsets will be saved in the `data` folder.

---

For further details on model implementation or dataset processing, refer to the accompanying paper.  
**Note**: Ensure all dependencies and environment configurations are properly set before execution.
