import numpy as np
import time
from multiprocessing import shared_memory


# class SharedMemoryQueue


class SharedMemoryWithIndexes:
    """
    A class that provides a queue-like mechanism using shared memory to store and retrieve fixed-size audio chunks.
    It uses shared memory for inter-process communication (IPC) and indexes to manage where data is written to and read from.

    Shared memory layout:
    - First byte: lock (0 or 1) to control access and prevent race conditions.
    - Next 128 bytes: an index array indicating which audio chunks are free (0 = free, 1 = occupied).
    - Next 128 bytes: an index array indicating which audio chunks have been read (0 = unread, 1 = read).
    - The rest is for storing audio chunks. Each chunk is a fixed size of 96000 bytes (for 48000 samples with 2 bytes per sample).
    """

    def __init__(self, memory_file_path):
        """
        Initialize shared memory and necessary indexes for read and write operations.

        :param memory_file_path: The name or path to the shared memory file.

        Memory Layout:
        - First 16 bytes: lock and indexes
        - Remaining bytes for audio data.
        - Total size of memory = 128 audio chunks * 48000 samples * 2 bytes/sample + 32 bytes (for metadata) = 12,288,258 bytes.
        """
        try:
            # Try attaching to existing shared memory
            self.memory = shared_memory.SharedMemory(name=memory_file_path)
        except:
            # If shared memory doesn't exist, create a new one
            self.memory = shared_memory.SharedMemory(
                name=memory_file_path, create=True, size=128 * 48000 * 2 + 32
            )

        # Initialize memory segments
        self.lock = np.frombuffer(
            self.memory.buf[0:1], dtype=np.uint8
        )  # Lock for safe access (1 byte)
        self.free_memory_index = np.frombuffer(
            self.memory.buf[1 : 1 + 128], dtype=np.uint8
        )  # Free memory index (128 bytes)
        self.read_index = np.frombuffer(
            self.memory.buf[2 + 128 : 2 + 128 * 2], dtype=np.uint8
        )  # Read index (128 bytes)
        self.item_size = 96000  # Fixed size for each audio chunk (96000 bytes)

    def acquire_lock(self):
        """
        Try to acquire the lock by setting the lock byte to 1.
        If the lock is already held, wait until it is released.
        """
        while self.lock[0] == 1:
            time.sleep(0.001)  # Wait until the lock is released
        self.lock[0] = 1  # Acquire the lock

    def release_lock(self):
        """
        Release the lock by setting the lock byte to 0.
        """
        self.lock[0] = 0

    def enqueue(self, data: np.ndarray):
        """
        Enqueue an audio chunk into the shared memory.

        This method writes the given `data` to the first available free memory chunk
        (as indicated by the `free_memory_index`).

        :param data: A NumPy array containing the audio chunk (size must match `self.item_size`).
        """
        self.acquire_lock()

        try:
            # Find the first available free memory index (where index is 0)
            first_index = np.where(self.free_memory_index == 0)[0]

            if len(first_index) == 0:
                raise OverflowError("No free memory slots available.")

            # Take the first free slot
            first_index = int(first_index[0])
            print(first_index)
            # Copy the data into the shared memory buffer
            arr_to_copy = np.ndarray(
                self.item_size,
                dtype=np.int16,
                buffer=self.memory.buf[
                    first_index * self.item_size * 2 : (first_index + 1)
                    * self.item_size
                    * 2
                ],
            )
            np.copyto(arr_to_copy, data)

            # Mark this chunk as occupied
            self.free_memory_index[first_index] = 1

        finally:
            # Always release the lock, even if an error occurs
            self.release_lock()

    def dequeue(self):
        """
        Dequeue an audio chunk from the shared memory.

        This method retrieves the first unread audio chunk (as indicated by the `read_index`).

        :return: A NumPy array containing the dequeued audio chunk.
        """
        self.acquire_lock()
        try:
            # Find the first unread memory index
            first_index = np.where(self.read_index == 0)[0]
            if len(first_index) == 0:
                print("No unread memory slots available.")
            first_index = int(first_index[0])
            print(first_index)
            # Retrieve the audio data from shared memory
            data = np.ndarray(
                self.item_size,
                dtype=np.int16,
                buffer=self.memory.buf[
                    first_index * self.item_size * 2 : (first_index + 1)
                    * self.item_size
                    * 2
                ],
            )
            # Mark this chunk as read
            self.read_index[first_index] = 1
            return data
        finally:
            self.release_lock()

    def update_indexes(self, read: bool):
        """
        Update the indexes in shared memory.

        :param read: If True, update the read indexes. If False, update the free memory indexes.
        """
        if read:
            # Update the shared memory with the current read index
            self.memory[1 : 1 + 128] = self.read_index
        else:
            # Update the shared memory with the current free memory index
            self.memory[1 + 128 : 1 + 128 * 2] = self.free_memory_index

    def close(self):
        """
        Close the shared memory.
        """
        self.memory.close()

    def delete(self):
        """
        Close and unlink the shared memory.

        This method ensures that the shared memory is closed and unlinked from the system,
        safely releasing the resources.
        """
        try:
            del self.free_memory_index
            del self.read_index
            del self.lock
        except Exception as e:
            print(f"Error cleaning up: {e}")
        self.memory.close()  # Close the shared memory object
        self.memory.unlink()  # Unlink the shared memory from the system
