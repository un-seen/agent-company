import threading
import time
from redis import Redis
import os
from typing import Union
from agentcompany.application.manager.toolkit import run_with as create_manager
from agentcompany.driver.models import OpenAIServerModel
import logfire

logfire.configure()

class RedisManager:
    
    def __init__(self, company_name: str):
        self.company_name = company_name
        self.user_input_queue_name = f"manager:user_input"
        # Create a Redis client. Setting decode_responses=True makes it return strings.
        self.redis_client = Redis.from_url(os.environ["REDIS_URL"])
        self._stop_event = threading.Event()
        # Create the worker thread as a daemon so that it dies when the main thread exits.
        self.model = OpenAIServerModel("gpt-4o-mini")
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe(self.company_name)
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

    def receive(self) -> Union[str, None]:
        """
        Receive a message from the Redis queue. This is a blocking operation.
        """
        
        content = self.pubsub.get_message()
        if not content:
            return None
        return bytes.decode(content)
        
        
    def _worker(self) -> None:
        """
        Worker thread method. It listens for messages on the Redis queue using a blocking pop
        (with a timeout) and processes any messages that arrive.
        """
        agent = create_manager(self.model, self.company_name)
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
