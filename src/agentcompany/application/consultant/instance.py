import threading
import time
from redis import Redis
import os
from typing import Union
from agentcompany.application.consultant.toolkit import run_with as create_manager
from agentcompany.driver.models import OpenAIServerModel
import logfire
from typing import Any

if os.environ["LOGFIRE_TOKEN"]:
    logfire.configure(token=os.environ["LOGFIRE_TOKEN"])



def default_sop(company_name: str) -> str:
    return f"""
    Standard Operating Procedure for {company_name}:
    
    1. Define strategy for the task at hand.
    2. Reason about the strategy to come up with a plan.
    3. Execute the plan and create a final answer.
    4. Get feedback on the final answer.
    5. Repeat steps 1-4 if feedback is not positive.
    """
class ConsultantApp:
    
    def __init__(self, company_name: str, sop: str = None):
        self.company_name = company_name
        self.sop = sop or default_sop(company_name)
        self.user_input_queue_name = f"user_input:{company_name}"
        self.agent_output_queue_name = f"{company_name}"
        # Create a Redis client. Setting decode_responses=True makes it return strings.
        self.redis_client = Redis.from_url(os.environ["REDIS_URL"])
        self._stop_event = threading.Event()
        # Create the worker thread as a daemon so that it dies when the main thread exits.
        self.model = OpenAIServerModel("gpt-4o-mini")
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        
    def start(self) -> None:
        """Starts the worker thread."""
        self.worker_thread.start()
        print(f"{self.company_name} thread started.")
        
    def send_user_input(self, message: str) -> bool:
        """
        Send a message to the Redis queue.
        """
        self.redis_client.rpush(self.user_input_queue_name, message)
        return True

    def get_channel_name(self) -> str:
        """
        Returns the channel name.
        """
        return self.company_name
        
    def _worker(self) -> None:
        """
        Worker thread method. It listens for messages on the Redis queue using a blocking pop
        (with a timeout) and processes any messages that arrive.
        """
        agent = create_manager(self.model, self.company_name, self.sop)
        while not self._stop_event.is_set():
            try:
                # blpop returns a tuple (queue_name, message) if a message is found, or None on timeout.
                message = self.redis_client.lpop(self.user_input_queue_name)
                if message:
                    message = bytes.decode(message)
                    # Process the message (here, we just print it)
                    print("Processing message:", message)
                    control_message = agent.run(message)
                    print("Control message:", control_message)
                    print("Process completed.")
                    # Delay to prevent busy-waiting
                    time.sleep(1)
            except Exception as e:
                print("Error in worker thread:", e)
        print(f"{self.company_name} thread stopping.")

    def stop(self) -> None:
        """Stops the worker thread gracefully."""
        self._stop_event.set()
        self.worker_thread.join()
        print("Worker thread stopped.")
