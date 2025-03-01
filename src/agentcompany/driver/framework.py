import abc
import threading
import time
import os
from redis import Redis
from agentcompany.driver import ManagerAgent
from agentcompany.driver.models import OpenAIServerModel
from agentcompany.driver.runner import FunctionRunner
import logfire
from typing import Any

# Configure logging if token is provided.
if os.environ.get("LOGFIRE_TOKEN"):
    logfire.configure(token=os.environ["LOGFIRE_TOKEN"])


class BasicWorkflow(abc.ABC):
    def __init__(self, workflow_name: str, model_name: str = "gpt-4o-mini"):
        self.workflow_name = workflow_name
        self.user_input_queue_name = f"user_input:{workflow_name}"
        self.workflow_output_queue_name = f"{workflow_name}"
        # Create a Redis client. decode_responses=True returns strings.
        self.redis_client = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
        self._stop_event = threading.Event()
        # Create the worker thread as a daemon so that it dies when the main thread exits.
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)

    def start(self) -> None:
        """Starts the worker thread."""
        self.worker_thread.start()
        print(f"{self.workflow_name} thread started.")

    def send_user_input(self, message: str) -> bool:
        """Sends a message to the Redis queue."""
        self.redis_client.rpush(self.user_input_queue_name, message)
        return True

    def get_channel_name(self) -> str:
        """Returns the channel name."""
        return self.workflow_name

    @abc.abstractmethod
    def create_runner(self) -> FunctionRunner:
        """
        Returns a ManagerAgent instance.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("create_manager_agent method must be implemented by subclasses.")

    def _worker(self) -> None:
        """
        Worker thread method. It listens for messages on the Redis queue (blocking pop)
        and processes any messages that arrive.
        """
        main_runner = self.create_runner()
        while not self._stop_event.is_set():
            try:
                # lpop returns a message or None.
                message = self.redis_client.lpop(self.user_input_queue_name)
                if message:
                    print("Processing message:", message)
                    control_message = main_runner.run(message)
                    self.redis_client.publish(self.workflow_output_queue_name, control_message)
                    print("Control message:", control_message)
                    print("Process completed.")
                    # Delay to prevent busy-waiting.
                    time.sleep(1)
            except Exception as e:
                print("Error in worker thread:", e)
        print(f"{self.workflow_name} thread stopping.")

    def stop(self) -> None:
        """Stops the worker thread gracefully."""
        self._stop_event.set()
        self.worker_thread.join()
        print("Worker thread stopped.")
        

class BasicCompany(abc.ABC):
    def __init__(self, company_name: str, model_name: str = "gpt-4o-mini"):
        self.company_name = company_name
        self.user_input_queue_name = f"user_input:{company_name}"
        self.agent_output_queue_name = f"{company_name}"
        # Create a Redis client. decode_responses=True returns strings.
        self.redis_client = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
        self._stop_event = threading.Event()
        # Initialize a model (you may later extend this logic)
        self.model = OpenAIServerModel(model_name)
        # Create the worker thread as a daemon so that it dies when the main thread exits.
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)

    def start(self) -> None:
        """Starts the worker thread."""
        self.worker_thread.start()
        print(f"{self.company_name} thread started.")

    def send_user_input(self, message: str) -> bool:
        """Sends a message to the Redis queue."""
        self.redis_client.rpush(self.user_input_queue_name, message)
        return True

    def get_channel_name(self) -> str:
        """Returns the channel name."""
        return self.company_name

    @abc.abstractmethod
    def create_manager_agent(self) -> ManagerAgent:
        """
        Returns a ManagerAgent instance.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("create_manager_agent method must be implemented by subclasses.")

    def _worker(self) -> None:
        """
        Worker thread method. It listens for messages on the Redis queue (blocking pop)
        and processes any messages that arrive.
        """
        manager_agent = self.create_manager_agent()
        while not self._stop_event.is_set():
            try:
                # lpop returns a message or None.
                message = self.redis_client.lpop(self.user_input_queue_name)
                if message:
                    print("Processing message:", message)
                    control_message = manager_agent.run(message)
                    print("Control message:", control_message)
                    print("Process completed.")
                    # Delay to prevent busy-waiting.
                    time.sleep(1)
            except Exception as e:
                print("Error in worker thread:", e)
        print(f"{self.company_name} thread stopping.")

    def stop(self) -> None:
        """Stops the worker thread gracefully."""
        self._stop_event.set()
        self.worker_thread.join()
        print("Worker thread stopped.")


__all__ = ["BasicCompany"]