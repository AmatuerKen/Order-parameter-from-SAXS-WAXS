# Order-parameter-from-SAXS-WAXS

These are python codes/notebooks to extract order parameters of liquid crystalline materials from 2D x-ray scattering pattern, either SAXS or WAXS depending on the geometry (q range). The raw data format (including metadata) is ESRF edf file to accommendate the DEXS equipment at UPenn LRSM. 


The order parameter is extracted using Kratky model. The model, average limit, and baseline substraction are implemented based on this paper: https://doi.org/10.1080/02678292.2018.1455227

The procedure is:
1. Creating qmap: it creates a correpondence between indices (x,y) and (q and angle), or (qx and qy).

   *qmap only needs to be created once for all data collected from the same experimental setup (sample to detector distance, beam center, etc)
        
3. Creating mask: it creates regions of interest for averaging intensity

   *mask only needs to be created when different regions of interest are needed.

5. Analysis: it extracts the average intensity in the mask and order parameters.

   *run the template to process one file as an example.
   *run the bulk to process one or all the edf files in directory, order parameters and fitting parameters are saved in a csv file.

Therefore, ideally functions_SAXS.py should be added to your PATH, and other notebooks should be in the same folder with the data folder, not the data files.
