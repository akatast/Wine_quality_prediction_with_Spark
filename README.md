# A wine quality prediction ML model in Spark over AWS

**GitHub URLs:**

Parallel training implementation-
https://github.com/akatast/Wine_quality_prediction_with_Spark/blob/main/wineQPredModelTraining.py

Single machine prediction application-
https://github.com/akatast/Wine_quality_prediction_with_Spark/blob/main/wineQPredModelValidation.py

**Docker hub URL:**

Docker container for prediction application-

https://hub.docker.com/repository/docker/as5721/as57dockerpublic  



**Command to execute docker container using input file-**

Please use following command to execute Docker container for prediction application using local input file:  
``sudo docker run -it -v `pwd`/TestDataset.csv:/dataset/TestDataset.csv as5721/as57dockerpublic:test-wine-qp /dataset/TestDataset.csv``

Above command is expecting TestDataset.csv as input file, in case of different input please change the file name in the command.  

Same is present on page-1 for attached PFD guide-  
https://github.com/akatast/Wine_quality_prediction/blob/main/Step-by-step%20guide.pdf


Please make sure docker is started and input file TestDataset.csv is available at the same directory where this command is being submitted.

Please check attached PDF document for step-by-step cloud environment set-up, model training and execution of prediction application.
