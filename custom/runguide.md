# With custom checkpoint
python sam3_mask_generator.py ./images "a building" --checkpoint ./models/sam3.pt

python .\tests\sam3_mask_generator.py .\tests\images\20_525636_347626_768\ "a building"


python tests/sam3_mask_generator.py ../test/geo_test_images/20_525636_347626_768/ "a building"
ls ../test/geo_test_images/20_525636_347626_768/

python demo.py ../test/geo_test_images/20_525636_347626_768/


scp -i eks.pem -r ubuntu@ec2-98-87-168-8.compute-1.amazonaws.com:~/test/geo_test_images/20_525636_347626_768/usd ./20_525636_347626_768/usd
aws s3 cp ~/test/geo_test_images s3://osm-data-export/sam-3d-objects/ --recursive
aws s3 sync ~/test/geo_test_images s3://osm-data-export/sam-3d-objects/



# Basic usage - process an image
python demo_full_image.py path/to/your/image.png

# Specify custom output directory
python demo_full_image.py path/to/your/image.png --output-dir ./output

# Skip mesh decoding (faster, but no USD export)
python demo_full_image.py path/to/your/image.png --no-mesh

# Custom USD scale factor
python demo_full_image.py path/to/your/image.png --usd-scale 50.0