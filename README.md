The assignment can be run using the .ipynb notebook located in the src folder. 

The notebook was designed for use in Google Colab, so in order to run the notebook as-is, open it in Colab, download the data folder into your google drive and rename it to Gaussian_Splat. 
This will allow for smooth access to necessary files after mounting google drive from colab.
Alternatively, the code in the Setup and Execution section can be adjusted to acdommodate other file systems.

If running the notebook with the intended setup as described above, start by running all the cells in the Setup section in order. 
Then run either the Execution section (for 2D Gaussian splatting) or the Execution 3D section. 
Within each section, code can be adjusted to load different pieces of data, set the number of Gaussians (N), and turn densification on or off.
Output images are saved to the Gaussian_Splat/Results folder in the user's google drive.
