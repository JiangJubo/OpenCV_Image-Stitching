# OpenCv_Image-Stitching
This is an image stitching project.

Step 1: Gray processing of images to be stitched
Step 2: Use Scale-invariant feature transform (SIFT) to find feature points in two images
Step 3: Use KnnMatch to match feature points
Step 4: Use Random Sample Consensus (RANSAC) to calculate transfer matrix H
Step 5: Use transfer matrix H stitch two images
