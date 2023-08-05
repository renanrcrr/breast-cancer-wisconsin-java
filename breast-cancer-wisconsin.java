import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.trees.RandomForest;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.AttributeSelection;

public class ComplexBreastCancerClassification {

    public static void main(String[] args) throws Exception {
        // Load the dataset
        DataSource source = new DataSource("path_to/breast-cancer-wisconsin.arff"); // Replace with the actual path
        Instances dataset = source.getDataSet();
        
        // Set class attribute
        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Perform attribute selection using CfsSubsetEval and GreedyStepwise
        AttributeSelection attributeSelection = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
        attributeSelection.setEvaluator(eval);
        attributeSelection.setSearch(search);
        attributeSelection.SelectAttributes(dataset);

        Instances selectedData = attributeSelection.reduceDimensionality(dataset);

        // Initialize and configure the Random Forest classifier
        RandomForest randomForest = new RandomForest();
        randomForest.setNumTrees(100);
        randomForest.setSeed(1);

        // Perform cross-validation with RandomSubSpace
        RandomSubSpace randomSubSpace = new RandomSubSpace();
        randomSubSpace.setClassifier(randomForest);
        randomSubSpace.setNumExecutionSlots(4);
        
        Evaluation evaluation = new Evaluation(selectedData);
        evaluation.crossValidateModel(randomSubSpace, selectedData, 10, new java.util.Random(1));

        // Print evaluation results
        System.out.println("===== Complex Evaluation Results =====");
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toClassDetailsString());
        System.out.println(evaluation.toMatrixString());
    }
}
