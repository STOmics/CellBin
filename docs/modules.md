CellBin allows software flexibility to support individual single-step module operations. 
The corresponding execution is placed in directory __scripts__.

## Image quality control
```text
python qc.py
--tiles_path /data/SS200000135TL_D1
--chip_no SS200000135TL_D1
```
* ```--tiles_path``` The path of all tiles. 
* ```--chip_no``` The chip number of the Stereo-seq data. 

## Stitching
```text
python stitching.py
--tiles_path /data/SS200000135TL_D1
--output_file /result/stitched_image.tif
```
* ```--tiles_path``` The path of all tiles. 
* ```--output_file``` The output path of the stitched image file. 

## Registration
```text
python registration.py
--image_file /result/stitched_image.tif
--output_file /result/registered_image.tif
--gene_exp_data /data/SS200000135TL_D1.gem.gz
--chip_no SS200000135TL_D1
```
* ```--image_file``` The stitched image.
* ```--output_file``` The registered image.
* ```--gene_exp_data``` The compressed file of spatial gene expression data.
* ```--chip_no``` The chip number of the Stereo-seq data. 

## Tissue segmentation
```text
python segmentation.py
--type tissue
--image_file /result/registered_image.tif
--output_file /result/tissue_mask.tif
```
* ```--type``` Can be tissue or nuclei. 
* ```--image_file```  The registered image. 
* ```--output_file``` The tissue mask image. 

## Nuclei segmentation
```text
python segmentation.py
--type nuclei
--image_file /result/registered_image.tif
--output_file /result/nuclei_mask.tif
```
* ```--type``` Can be tissue or nuclei. 
* ```--image_file``` The registered image.
* ```--output_file``` The nuclei mask image. 

## Nuclei mask filtering
```text
python filtering.py
--tissue_mask /result/tissue_mask.tif
--nuclei_mask /result/nuclei_mask.tif
--output_file /result/nuclei_mask.tif
```
* ```--tissue_mask``` The tissue mask image. 
* ```--nuclei_mask``` The nuclei mask image. 
* ```--output_file``` The filtered nuclei mask image. 

## Molecule labeling
```text
python labeling.py
--image_file /result/nuclei_mask.tif
--gene_exp_data /data/SS200000135TL_D1.gem.gz
--output_path /result
```
* ```--image_file``` The nuclei mask image. 
* ```--gene_exp_data``` The compressed file of spatial gene expression data. 
* ```--output_path``` The output path. 

