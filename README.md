![HACA3 features](figures/GA.png)

# HACA3: A unified approach for multi-site MR image harmonization | [Paper](https://www.sciencedirect.com/science/article/pii/S0895611123001039)

HACA3 is an advanced approach for multi-site MRI harmonization. This page provides a gentle introduction to HACA3 inference and training. 

- Publication: [Zuo et al. HACA3: A unified approach for multi-site MR image harmonization. *Computerized Medical Imaging 
and Graphics, 2023.*](https://www.sciencedirect.com/science/article/pii/S0895611123001039)

- Citation:    
    ```bibtex
    @article{ZUO2023102285,
    title = {HACA3: A unified approach for multi-site MR image harmonization},
    journal = {Computerized Medical Imaging and Graphics},
    volume = {109},
    pages = {102285},
    year = {2023},
    issn = {0895-6111},
    doi = {https://doi.org/10.1016/j.compmedimag.2023.102285},
    author = {Lianrui Zuo and Yihao Liu and Yuan Xue and Blake E. Dewey and
              Samuel W. Remedios and Savannah P. Hays and Murat Bilgel and 
              Ellen M. Mowry and Scott D. Newsome and Peter A. Calabresi and 
              Susan M. Resnick and Jerry L. Prince and Aaron Carass}
    }
    ```

## 1. Introduction and motivation
### 1.1 The flexibility and variability in MRI
The flexibility of magnetic resonance imaging (MRI) has enabled multiple tissue contrasts to be routinely acquired in a single 
imaging session. For example, T1-weighted, T2-weighed, and FLAIR images are often acquired to better reveal different 
tissue properties. However, this flexibility of MRI also introduces 
drawbacks, most notably the ***lack of standardization and consistency*** between imaging studies. Various factors can 
cause undesired contrast variations in the acquired images. These factors include but not limited to
- Pulse sequences, e.g., MPRAGE, SPGR
- Imaging parameters, e.g., flip angle, TE
- Scanner manufacturers, e.g, Siemens, GE
- Technicians and site preference. 

### 1.2 Why should we harmonize MR images?
These contrast variations sometimes can be subtle, but often not negligible, especially in ***multi-site*** and 
***longitudinal*** studies. 
- ***Example #1***: multi-site inconsistency
    ![multi-site harmonization](figures/multi_site.png)

- ***Example #2***: longitudinal study
    ![longitudinal harmonization](figures/longitudinal.png)
## 2. Prerequisites 
Standard neuroimage preprocessing steps are needed before running HACA3. These preprocessing steps include:
- Inhomogeneity correction
- Registration to MNI space (1mm isotropic resolution)
- Super-resolution for 2D acquired scans. This step is optional, but recommended for optimal performance. 
See [SMORE](https://github.com/volcanofly/SMORE-Super-resolution-for-3D-medical-images-MRI) for more details.

## 3. Installation and pretrained weights

### 3.1 Option 1 (recommended): Run HACA3 through singularity image
In general, no installation of HACA3 is required with this option. 
Singularity image of HACA3 model can be directly downloaded [**here**](https://iacl.ece.jhu.edu/~lianrui/haca3/haca3_main.sif).


### 3.2 Option 2: Install from source using `pip`
1. Clone the repository:
    ```bash
    git clone https://github.com/lianruizuo/haca3.git 
    ```
2. Navigate to the directory:
    ```bash
    cd haca3
    ```
3. Install dependencies:
    ```bash
    pip install . 
    ```
Package requirements are automatically handled. To see a list of requirements, see `setup.py` L50-60. 
This installs the `haca3` package and creates two CLI aliases `haca3-train` and `haca3-test`.


### 3.3 Pretrained weights
Pretrained weights of HACA3 can be downloaded [**here**](https://iacl.ece.jhu.edu/~lianrui/haca3/harmonization_public.pt). 
This model was trained on public datasets including the structural MR images from [IXI](https://brain-development.org/ixi-dataset/), 
[OASIS3](https://www.oasis-brains.org), and [BLSA](https://www.nia.nih.gov/research/labs/blsa) dataset.
HACA3 uses a 3D convolutional network to combine multi-orientation 2D slices into a single 3D volume. 
Pretrained fusion model can be downloaded [**here**](https://iacl.ece.jhu.edu/~lianrui/haca3/fusion.pt).

## 4. Usage: Inference

### 4.1 Option 1 (recommended): Run HACA3 through singularity image
   ```bash
   singularity exec --nv -e haca3.sif haca3-test \
   --in-path [PATH-TO-INPUT-SOURCE-IMAGE-1] \
   --in-path [PATH-TO-INPUT-SOURCE-IMAGE-2, IF THERE ARE MULTIPLE SOURCE IMAGES] \
   --target-image [TARGET-IMAGE] \
   --harmonization-model [PRETRAINED-HACA3-MODEL] \
   --fusion-model [PRETRAINED-FUSION-MODEL] \
   --out-path [PATH-TO-HARMONIZED-IMAGE] \
   --intermediate-out-dir [DIRECTORY SAVES INTERMEDIATE RESULTS] 
   ```

- ***Example #3:***
    Suppose the task is to harmonize MR images from `Site A` to match the contrast of a pre-selected T1w image of 
    `Site B`. As a source site, `Site A` has T1w, T2w, and FLAIR images. The files are saved like this:
    ```
    ├──data_directory
        ├──site_A_t1w.nii.gz
        ├──site_A_t2w.nii.gz
        ├──site_A_flair.nii.gz
        └──site_B_t1w.nii.gz
    ```
    You can always retrain HACA3 using your own datasets. In this example, we choose to use the pretrained HACA3 weights 
    `harmonization.pt` and fusion model weights `fusion.pt` (see [3.3 Pretrained weights](#33-pretrained-weights) for 
    how to download these weights). The singularity command to run HACA3 is:
    ```bash
       singularity exec --nv -e haca3.sif haca3-test \
       --in-path data_directory/site_A_t1w.nii.gz \
       --in-path data_directory/site_A_t2w.nii.gz \
       --in-path data_directory/site_A_flair.nii.gz \
       --target-image data_directory/site_B_flair.nii.gz \
       --harmonization-model harmonization.pt \
       --fusion-model fusion.pt \
       --out-path output_directory/site_A_harmonized_to_site_B_t1w.nii.gz \
       --intermediate-out-dir output_directory
    ```
    The harmonized image and intermediate results will be saved at `output_directory`.


### 4.2 Option 2: Run HACA3 from source after installation
   ```bash
   haca3-test \
   --in-path [PATH-TO-INPUT-SOURCE-IMAGE-1] \
   --in-path [PATH-TO-INPUT-SOURCE-IMAGE-2, IF THERE ARE MULTIPLE SOURCE IMAGES] \
   --target-image [TARGET-IMAGE] \
   --harmonization-model [PRETRAINED-HACA3-MODEL] \
   --fusion-model [PRETRAINED-FUSION-MODEL] \
   --out-path [PATH-TO-HARMONIZED-IMAGE] \
   --intermediate-out-dir [DIRECTORY-THAT-SAVES-INTERMEDIATE-RESULTS] 
   ```


### 4.3 All options for inference
- ```--in-path```: file path to input source image. Multiple ```--in-path``` may be provided if there are multiple 
source images. See the above example for more details.
- ```--target-image```: file path to target image. HACA3 will match the contrast of source images to this target image.
- ```--target-theta```: In [HACA3](https://www.sciencedirect.com/science/article/pii/S0895611123001039), ```theta``` 
is a two-dimensional representation of image contrast. Target image contrast can be directly specified by providing 
a ```theta``` value, e.g., ```--target-theta 0.5 0.5```. Note: either ```--target-image``` or ```--target-image``` must 
be provided during inference. If both are provided, only ```--target-theta``` will be used.
- ```--norm-val```: normalization value. 
- ```--out-path```: file path to harmonized image. 
- ```--harmonization-model```: pretrained HACA3 weights. Pretrained model weights on IXI, OASIS and HCP data can 
be downloaded [here](https://iacl.ece.jhu.edu/~lianrui/haca3/harmonization_public.pt).
- ```--fusion-model```: pretrained fusion model weights. HACA3 uses a 3D convolutional network to combine multi-orientation
2D slices into a single 3D volume. Pretrained fusion model can be downloaded [here](https://iacl.ece.jhu.edu/~lianrui/haca3/fusion.pt).
- ```--save-intermediate```: if specified, intermediate results will be saved. Default: ```False```. Action: ```store_true```.
- ```--intermediate-out-dir```: directory to save intermediate results.
- ```--gpu-id```: integer number specifies which GPU to run HACA3.
- ```--num-batches```: During inference, HACA3 takes entire 3D MRI volumes as input. This may cause a considerable amount 
GPU memory. For reduced GPU memory consumption, source images maybe divided into smaller batches. 
However, this may slightly increase the inference time.


## 5. Acknowledgements

Special thanks to Samuel Remedios, Blake Dewey, and Yihao Liu for their feedbacks on HACA3 code release and this GitHub page.

The authors thank BLSA participants, as well as colleagues of the Laboratory of Behavioral Neuroscience (LBN) of NIA and 
the Image Analysis and Communications Laboratory (IACL) of JHU. 
This work was supported in part by the Intramural Research Program of the National Institutes of Health, 
National Institute on Aging, 
in part by the TREAT-MS study funded by the Patient-Centered Outcomes Research Institute (PCORI) grant MS-1610-37115 
(Co-PIs: Drs. S.D. Newsome and E.M. Mowry), 
in part by the National Science Foundation Graduate Research Fellowship under Grant No. DGE-1746891, 
in part by the NIH grant (R01NS082347, PI: P. Calabresi), National Multiple Sclerosis Society grant (RG-1907-34570, PI: D. Pham), 
and the DOD/Congressionally Directed Medical Research Programs (CDMRP) grant (MS190131, PI: J. Prince).