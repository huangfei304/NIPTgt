# NIPTgt

## Brief description

Non-invasive prenatal diagnosis (NIPD) can identify monogenic diseases early during pregnancy with negligible risk to fetus or mother. 

NIPTgt is a program to predict the mother genotype with the maternal blood and fetal genotype with the machine learning method.

## Basic Principles of NIPTgt

1)  Calling variants with the big NIPT sequencing data.
2)   Extracting features from the NIPT variants and true genotype of fetus and mother.
3)   Constructing model using the features and the true genotype of fetus and mother.  
4)  Using the constructed model and NIPT variant features to predict the fetus and mother genotype.
4)  Using the multiple machine learning model voting mechanisms to obtain the final results.

## NIPTgt result

1. Performance of the mother variant genotype.

   <img src="/data/mother_snp.png" alt="mother_snp" style="zoom:60%;" />

   

2.  performance of the fetus variant genotype.

   ```
   sampleID	total_snp	correct_snp	rate(%)
   test1	2538572	1969308	77.58
   test2	2305263	1695423	73.55
   test3	2261096	1557465	68.88
   ```

​	

