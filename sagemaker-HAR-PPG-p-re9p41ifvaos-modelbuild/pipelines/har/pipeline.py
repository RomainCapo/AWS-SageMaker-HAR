"""Example workflow pipeline script for HAR pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import logging
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.tensorflow import TensorFlow
from sagemaker.debugger import Rule, ProfilerRule, rule_configs

from botocore.exceptions import ClientError


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)

def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def resolve_ecr_uri_from_image_versions(sagemaker_session, image_versions, image_name):
    """ Gets ECR URI from image versions
    Args:
        sagemaker_session: boto3 session for sagemaker client
        image_versions: list of the image versions
        image_name: Name of the image

    Returns:
        ECR URI of the image version
    """

    #Fetch image details to get the Base Image URI
    for image_version in image_versions:
        if image_version['ImageVersionStatus'] == 'CREATED':
            image_arn = image_version["ImageVersionArn"]
            version = image_version["Version"]
            logger.info(f"Identified the latest image version: {image_arn}")
            response = sagemaker_session.sagemaker_client.describe_image_version(
                ImageName=image_name,
                Version=version
            )
            return response['ContainerImage']
    return None

def resolve_ecr_uri(sagemaker_session, image_arn):
    """Gets the ECR URI from the image name

    Args:
        sagemaker_session: boto3 session for sagemaker client
        image_name: name of the image

    Returns:
        ECR URI of the latest image version
    """

    # Fetching image name from image_arn (^arn:aws(-[\w]+)*:sagemaker:.+:[0-9]{12}:image/[a-z0-9]([-.]?[a-z0-9])*$)
    image_name = image_arn.partition("image/")[2]
    try:
        # Fetch the image versions
        next_token=''
        while True:
            response = sagemaker_session.sagemaker_client.list_image_versions(
                ImageName=image_name,
                MaxResults=100,
                SortBy='VERSION',
                SortOrder='DESCENDING',
                NextToken=next_token
            )
            ecr_uri = resolve_ecr_uri_from_image_versions(sagemaker_session, response['ImageVersions'], image_name)
            if "NextToken" in response:
                next_token = response["NextToken"]

            if ecr_uri is not None:
                return ecr_uri

        # Return error if no versions of the image found
        error_message = (
            f"No image version found for image name: {image_name}"
            )
        logger.error(error_message)
        raise Exception(error_message)

    except (ClientError, sagemaker_session.sagemaker_client.exceptions.ResourceNotFound) as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)

def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="HARPackageGroup",
    pipeline_name="HARPipeline",
    base_job_prefix="HAR",
    project_id="SageMakerProjectId"
):
    """Gets a SageMaker ML Pipeline instance working with on har data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline 
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.g4dn.xlarge")
    inference_instance_type = ParameterString(name="InferenceInstanceType", default_value="ml.m5.xlarge")
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    
    input_data = ParameterString(name="InputDataUrl", default_value=f"s3://sagemaker-input-data/PPG_ACC_dataset.zip")
    data_version = ParameterString(name="DataVersion", default_value="latest")
    
    window = ParameterInteger(name="Window", default_value=1200)
    overlap = ParameterInteger(name="Overlap", default_value=600)
    train_subjects = ParameterString(name="TrainSubjects", default_value="0-1-2-3")
    validation_subjects = ParameterString(name="ValidationSubjects", default_value="4")
    test_subjects = ParameterString(name="TestSubjects", default_value="5-6")
    
    classes = ParameterString(name="Classes", default_value="rest-squat-step")

    print(f"{str(overlap.default_value)}")
    
    epochs = ParameterInteger(name="Epochs", default_value=50)
    learning_rate = ParameterFloat(name="LearningRate", default_value=0.01)
    num_cell_dense1 = ParameterInteger(name="NumCellDense1", default_value=32)
    num_cell_lstm1 = ParameterInteger(name="NumCellLSTM1", default_value=32)
    num_cell_lstm2 = ParameterInteger(name="NumCellLSTM2", default_value=32)
    num_cell_lstm3 = ParameterInteger(name="NumCellLSTM3", default_value=32)
    dropout_rate = ParameterFloat(name="DropoutRate",default_value=0.5)
    
    processing_image_uri = sagemaker.image_uris.retrieve(framework="sklearn",
                                                         region=region,
                                                         version="0.20.0",
                                                         py_version="py3",
                                                         instance_type=processing_instance_type,)
        
    script_processor = ScriptProcessor(
        image_uri=processing_image_uri,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-har-preprocess",
        command=["python3"],
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    
    step_process = ProcessingStep(
        name="PreprocessHARData",
        processor=script_processor,
        outputs=[
            ProcessingOutput(output_name="train_val", source="/opt/ml/processing/train_val"),
            ProcessingOutput(output_name="subjects", source="/opt/ml/processing/subjects"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--input-data", input_data, 
                       "--window", str(window.default_value), 
                       "--overlap", str(overlap.default_value),
                       "--data_version", data_version,
                       "--train_subjects", train_subjects, 
                       "--validation_subjects",validation_subjects,
                       "--test_subjects", test_subjects,
                       "--classes", classes]
    )

    # training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/HARTrain"

    training_image_uri = sagemaker.image_uris.retrieve(framework="tensorflow", region=region, version="2.5", py_version="py37", instance_type=training_instance_type, image_scope="training")
        
    hyperparameters = {
        "epochs":epochs.default_value,
        "learning_rate":learning_rate.default_value,
        "num_cell_dense1":num_cell_dense1.default_value,
        "num_cell_lstm1":num_cell_lstm1.default_value,
        "num_cell_lstm2":num_cell_lstm2.default_value,
        "num_cell_lstm3":num_cell_lstm3.default_value,
        "dropout_rate":dropout_rate.default_value,
        "window":window.default_value,
        "overlap":overlap.default_value,
        "train_subjects":train_subjects,
        "validation_subjects":validation_subjects,
        "test_subjects":test_subjects,
        "classes":classes
    }
    
    built_in_rules = [
        Rule.sagemaker(rule_configs.overfit()),
        Rule.sagemaker(rule_configs.exploding_tensor()),
        Rule.sagemaker(rule_configs.loss_not_decreasing()),
        Rule.sagemaker(rule_configs.poor_weight_initialization()),
        Rule.sagemaker(rule_configs.saturated_activation()),
        Rule.sagemaker(rule_configs.vanishing_gradient()),
        Rule.sagemaker(rule_configs.overtraining()),
        Rule.sagemaker(rule_configs.confusion()),
        ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
    ]
    
    estimator_rnn = TensorFlow(
      entry_point=os.path.join(BASE_DIR, "rnn.py"),             # Your entry script
      role=role,
      framework_version="2.5",    # TensorFlow's version
      py_version="py37",
      instance_count=1,                   # "The number of GPUs instances to use"
      instance_type=training_instance_type,
      output_path=model_path,
      model_dir=False,
      sagemaker_session=sagemaker_session,
      hyperparameters=hyperparameters,
      rules=built_in_rules,
      metric_definitions=[
            {'Name':'train:loss', 'Regex':'loss: ([0-9\.]+)'},
            {'Name':'train:accuracy', 'Regex':'accuracy: ([0-9\.]+)'},
            {'Name':'test:loss', 'Regex':'val_loss: ([0-9\.]+)'},
            {'Name':'test:accuracy', 'Regex':'val_accuracy: ([0-9\.]+)'}
        ],
      enable_sagemaker_metrics=False,
    )
    
    step_train = TrainingStep(
        name="TrainHARModel",
        estimator=estimator_rnn,
        inputs={
            "train_val": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train_val"].S3Output.S3Uri,),
            "subjects": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["subjects"].S3Output.S3Uri,)
        },
    )

    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=training_image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-abalone-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    evaluation_report = PropertyFile(
        name="HAREvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateHARModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts,destination="/opt/ml/processing/model",),
            ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,destination="/opt/ml/processing/test",),],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )

    inference_image_uri = sagemaker.image_uris.retrieve(framework="tensorflow",region=region,version="2.5",py_version="py37",instance_type=inference_instance_type,image_scope="inference")
    
    step_register = RegisterModel(
        name="RegisterHARModel",
        estimator=estimator_rnn,
        image_uri=inference_image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv", "application/json"],
        response_types=["text/csv", "application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # condition step for evaluating model quality and branching execution
    cont_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="multiclass_classification_metrics.accuracy.value"
        ),
        right=0.8,
    )

    step_cond = ConditionStep(
        name="CheckAccuracyHAREvaluation",
        conditions=[cont_gte],
        if_steps=[step_register],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            processing_instance_type,
            training_instance_type,
            inference_instance_type,
            model_approval_status,
            input_data,
            data_version,
            window,
            overlap,
            train_subjects,
            validation_subjects,
            test_subjects,
            epochs,
            learning_rate,
            num_cell_dense1,
            num_cell_lstm1,
            num_cell_lstm2,
            num_cell_lstm3,
            dropout_rate,
            classes
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )
    return pipeline