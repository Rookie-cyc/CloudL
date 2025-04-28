import random
import time
import logging
import mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_random_data():
    try:
        with mlflow.start_run():
            mlflow.log_params({'app_name': 'MyApp', 'version': '1.0'})
            start_time = time.time()

            while True:
                # Generate random data
                data = random.randint(0, 100)
                
                # Log data using standard logging
                logger.info(f"Generated random data: {data}")
                
                # Log data as metric to MLflow
                mlflow.log_metric("my_metric", data)
                
                # Calculate and log uptime
                uptime = time.time() - start_time
                mlflow.log_metric("app_uptime_seconds", uptime)
                
                time.sleep(2)

    except KeyboardInterrupt:
        logger.info("Monitoring stopped due to KeyboardInterrupt.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    logger.info("Monitoring and Logging Application Started.")
    try:
        generate_random_data()
    except Exception as e:
        logger.error(f"Application crashed with error: {str(e)}")
    logger.info("Monitoring and Logging Application Stopped.")
