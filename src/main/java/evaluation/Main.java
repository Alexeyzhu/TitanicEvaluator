package evaluation;

import tech.tablesaw.api.Table;
import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        String testPath = "src/main/java/evaluation/test.csv"; // Path to test file in csv format
        String modelPath = "src/main/java/evaluation/modelTrain/lr.pmml"; // Path to trained model in PMML format
        String submissionPath = "submission.csv";   // Path where results should be saved in csv format

        ModelEvaluator evaluator = new ModelEvaluator(modelPath); // Load model
        Table survivalPrediction = evaluator.predict(testPath);   // Predict result on test data

        try {
            survivalPrediction.write().csv(submissionPath);
            System.out.println(String.format("File %s successfully saved", submissionPath));
        } catch (IOException e) {
            System.out.println(String.format("Failed to save file %s", submissionPath));
        }catch(NullPointerException e){
            System.out.println("Prediction table is empty");
        }
    }


}
