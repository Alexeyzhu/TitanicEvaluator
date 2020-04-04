package evaluation;

import com.google.common.collect.ImmutableMap;
import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.*;
import org.xml.sax.SAXException;
import tech.tablesaw.api.*;

import javax.xml.bind.JAXBException;
import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ModelEvaluator {

    Evaluator evaluator; // Loaded model

    /**
     * Loads trained model from PMML file
     *
     * @param modelPath Path to trained model in PMML format
     */
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

    /**
     * Predicts survival chance for people on the Titanic from test file
     *
     * @param testPath Path to test file in csv format
     * @return Table (PassengerID, Survived) with results of prediction in Survived column
     */
    public Table predict(String testPath) {
        Table testTitanicTable;

        // reading csv file with test data
        try {
            testTitanicTable = Table.read().csv(testPath);
        } catch (IllegalStateException e) {
            System.out.println(String.format("Cannot find file %s", testPath));
            return null;
        } catch (IOException e) {
            System.out.println(String.format("Failed to read file %s", testPath));
            return null;
        }

        if (this.evaluator == null) {
            return null;
        }

        IntColumn passengerIdColumn = (IntColumn) testTitanicTable.column("PassengerId");
        Table testTable = preprocessTest(testTitanicTable);

        // Printing input (x1, x2, .., xn) fields
        List<? extends InputField> inputFields = this.evaluator.getInputFields();

        // Resulting table for submission
        IntColumn survivalPrediction = IntColumn.create("Survived");

        // Iterating through columnar data to predict
        for (Row testRow : testTable) {

            Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();

            // Mapping the record field-by-field from data source schema to PMML schema
            for (InputField inputField : inputFields) {
                FieldName inputName = inputField.getName();
                Object rawValue;

                ColumnType columnType = testRow.getColumnType(inputName.getValue());
                if (columnType.equals(ColumnType.DOUBLE)) {
                    rawValue = testRow.getDouble(inputName.getValue());
                } else {
                    rawValue = testRow.getInt(inputName.getValue());
                }

                // Transforming an arbitrary user-supplied value to a known-good PMML value
                FieldValue inputValue = inputField.prepare(rawValue);

                arguments.put(inputName, inputValue);
            }

            // Evaluating the model with known-good arguments
            Map<FieldName, ?> results = evaluator.evaluate(arguments);

            // Decoupling results from the JPMML-Evaluator runtime environment
            Map<String, ?> resultRecord = EvaluatorUtil.decodeAll(results);

            // Gathering result of prediction
            survivalPrediction.append((Integer) resultRecord.get("Survived"));
        }

        Table submissionTable = Table.create("submission");
        submissionTable.addColumns(passengerIdColumn, survivalPrediction);

        return submissionTable;
    }

    /**
     * Preprocess test data - bin age and fare columns, map sex and embarked columns, remove unused columns
     *
     * @param testTitanicTable Table with test data to preprocess
     * @return Preprocessed table with test data
     */
    private Table preprocessTest(Table testTitanicTable) {
        // Remove unused columns
        testTitanicTable.removeColumns("PassengerId", "Ticket", "Cabin", "Name");

        DoubleColumn ageColumn = testTitanicTable.doubleColumn("Age");
        double ageMean = ageColumn.mean();
        // Fill missing age values with mean
        ageColumn.set(ageColumn.isMissing(), ageMean);

        // thresholds for bins taken from method pandas.qcut on training set
        Double[] ageBins = new Double[]{0.0, 18.0, 24.0, 29.0, 35.0, 42.0, 80.10};
        DoubleColumn ageBinned = binColumn(ageColumn, ageBins, "Age");

        // Replace non-binned column with binned one
        testTitanicTable.removeColumns("Age");
        testTitanicTable.addColumns(ageBinned);

        DoubleColumn fareColumn = testTitanicTable.doubleColumn("Fare");
        // Fill missing fare values with 0
        fareColumn.set(fareColumn.isMissing(), 0.0);

        // thresholds for bins taken from method pandas.qcut on training set
        Double[] fareBins = new Double[]{0.0, 7.775, 8.6625, 14.4542, 26.0, 52.36946667, 512.3293};
        DoubleColumn fareBinned = binColumn(fareColumn, fareBins, "Fare");

        // Replace non-binned column with binned one
        testTitanicTable.removeColumns("Fare");
        testTitanicTable.addColumns(fareBinned);

        StringColumn sexColumn = testTitanicTable.stringColumn("Sex");

        // Map string values to corresponding double values in column
        Map<String, Double> sexMapping = ImmutableMap.of("male", 0.0, "female", 1.0);
        DoubleColumn sexMapped = mapColumn(sexColumn, sexMapping, "Sex");

        // Replace string column with double (mapped) one
        testTitanicTable.removeColumns("Sex");
        testTitanicTable.addColumns(sexMapped);

        StringColumn embarkedColumn = testTitanicTable.stringColumn("Embarked");

        // Map string values to corresponding double values in column
        Map<String, Double> embarkedMapping = ImmutableMap.of("C", 0.0, "S", 1.0, "Q", 2.0);
        DoubleColumn embarkedMapped = mapColumn(embarkedColumn, embarkedMapping, "Embarked");

        // Replace string column with double (mapped) one
        testTitanicTable.removeColumns("Embarked");
        testTitanicTable.addColumns(embarkedMapped);

        return testTitanicTable;
    }

    /**
     * Map double values of a column to corresponding bin order numbers
     * If value lies inside bin range, then it mapped to order number of the bin
     *
     * @param columnToBin DoubleColumn whose values are required to bin
     * @param bins        Bin ranges
     * @param columnName  Name of result column
     * @return DoubleColumn with mapped values
     */
    private DoubleColumn binColumn(DoubleColumn columnToBin, Double[] bins, String columnName) {
        DoubleColumn columnBinned = DoubleColumn.create(columnName);

        for (Double rowValue : columnToBin) {
            for (int i = 0; i < bins.length - 1; i++) {
                // If value lies inside bin range, then it mapped to order number of the bin
                if (rowValue >= bins[i] && rowValue < bins[i + 1]) {
                    columnBinned.append((double) i);
                }
            }
        }

        return columnBinned;
    }

    /**
     * Map string values of a column to corresponding double values
     *
     * @param columnToMap StringColumn whose values are required to map
     * @param map         Mapping of string and double values
     * @param columnName  Name of result column
     * @return DoubleColumn with mapped values
     */
    private DoubleColumn mapColumn(StringColumn columnToMap, Map<String, Double> map, String columnName) {
        DoubleColumn columnMapped = DoubleColumn.create(columnName);

        for (String rowValue : columnToMap) {
            for (String key : map.keySet()) {
                if (key.equals(rowValue)) {
                    columnMapped.append(map.get(key));
                }
            }
        }

        return columnMapped;
    }
}
