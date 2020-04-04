# TitanicEvaluator

This repository is a solution for [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview) 
competition. It is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck. 

Project consist of 2 main parts:

* Python code that train the model on the **train.csv** input data and upload the trained model to a file **lr.pmml**
* The Java code that read the model from **lr.pmml** file, apply it to the test data **test.csv**, and upload the result
 to the **submission.csv** file. 
 
I used LogisticRegression model from sklearn library in Python, wrap it into [PMMLPipeline](https://github.com/jpmml/sklearn2pmml) 
and transferred it to Java via [JPMML](https://github.com/jpmml/jpmml-sklearn) library.

* Python part (*ModelTrain.py*)
```python
    from sklearn.linear_model import LogisticRegression
    from sklearn2pmml import sklearn2pmml
    from sklearn2pmml.pipeline import PMMLPipeline


    logreg = LogisticRegression(n_jobs=cores, solver='lbfgs', multi_class='multinomial')
    
    pipeline = PMMLPipeline([
        ("classifier", logreg)
    ])

    pipeline.fit(X_train, y_train)

    sklearn2pmml(pipeline, filename, with_repr=True)
```
 * Java part (*ModelEvaluator.java*)
```java
    import org.jpmml.evaluator.*;

    public ModelEvaluator(String modelPath) {
    
            try {
                // Loading and building a model evaluator from a PMML file
                this.evaluator = new LoadingModelEvaluatorBuilder().load(new File(modelPath)).build();
    
                // Performing the self-check for stability
                this.evaluator.verify();
    
            } catch (JAXBException | SAXException e) {
                System.out.println(String.format("Failed to load model %s", modelPath));
            } catch (IOException e) {
                System.out.println(String.format("Cannot find file %s", modelPath));
            }
    
        }
```
### How to run?

#### Prerequisites
* [Python](https://www.python.org/downloads/)  3.6 and higher 
* [Java JDK](https://www.oracle.com/java/technologies/javase-downloads.html)  8 and higher 
* [PyCharm](https://www.jetbrains.com/pycharm/)
* [IntellijIdea](https://www.jetbrains.com/idea/)

#### Clone project and setup

* Copy URL of repository -> Open IntellijIdea -> Get from Version Control -> Paste URL -> Open project
* Open subfolder **src/main/java/evaluation/modelTrain** in Pycharm 
* Open terminal in PyCharm and install **requirements.txt**
```bash
   pip install -r requirements.txt
```

#### Run program

* Run **ModelTrain.py** in PyCharm. It should create **lr.pmml** in modelTrain directory
* Rum **Main.java** in IntellijIdea. It shoulf create **submission.csv** in root directory
