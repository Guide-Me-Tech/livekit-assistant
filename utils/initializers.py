import chromadb
import json
import orjson
from chromadb.utils import embedding_functions
from chromadb.db.base import UniqueConstraintError
import dotenv
import utils.printing


def load_envs():
    dotenv.load_dotenv(verbose=True)


class OpenedActionsFormatter:
    def __init__(
        self,
    ):
        self.collection = None

    def GetChroma(self, username):
        chroma = chromadb.PersistentClient("./user_files/chromadb")
        collection_name = f"{username}_actions_collection"
        print(collection_name)
        # try:
        collection = chroma.get_collection(collection_name)
        # except:
        #     raise Exception(
        #         f"Collection {collection_name} not found --- please train actions first"
        #     )
        # self.collection = collection

    def Query(self, query_string):
        results = self.collection.query(query_texts=[query_string], n_results=1)
        return results

    def Format(self, results):
        results = results["documents"][0]
        results = orjson.dumps(results)
        return results

    def FormatToGeneralAnswer(self, results, user_message):
        print(results)
        o = orjson.loads(results["documents"][0][0])
        message, response = format_actions_sequence(o, user_message=user_message)
        if "action" in message.content[:10]:
            return orjson.loads(message.content[8:]), response
        return orjson.loads(message.content), response


class TrainActionsBotv2:
    def __init__(self):
        self.actions = []
        self.username = None

    def SetActions(self, actions):
        self.actions = actions

    def AddAction(self, action):
        print("Got action", action)
        self.actions.append(action)

    def SetUsername(self, username):
        self.username = username

    def TrainandSave(self, sentence_transformer_ef):
        chroma_client = chromadb.PersistentClient("./user_files/chromadb")
        # need to change the database initilization process, so that it is not created for every user but only once
        # also need the change collection name from demo_collection to somethings else {XXXXXXXXXXXXXXXXXXXX}
        collection_name = self.username + "_actions_collection"
        try:
            collection = chroma_client.create_collection(
                collection_name,
                embedding_function=sentence_transformer_ef,
            )
        except UniqueConstraintError:
            print("UniqueConstraintError error occured - making new collection")
            chroma_client.delete_collection(collection_name)
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef,
            )
            print("Deleted data from collection")
        print("Added data to collection")
        # printing.printorange(self.actions)
        ids = [str(i) for i in range(len(self.actions))]
        collection.add(documents=[json.dumps(i) for i in self.actions], ids=ids)
        return "done"


class TrainActionsBot:
    def __init__(self):
        self.actions = []
        self.username = None

    def SetActions(self, actions):
        self.actions = actions

    def AddAction(self, action):
        print("Got action", action)
        self.actions.append(action)

    def SetUsername(self, username):
        self.username = username

    def TrainandSave(self, sentence_transformer_ef):
        chroma_client = chromadb.PersistentClient("./user_files/chromadb")
        # need to change the database initilization process, so that it is not created for every user but only once
        # also need the change collection name from demo_collection to somethings else {XXXXXXXXXXXXXXXXXXXX}
        collection_name = self.username + "_actions_collection"
        print("Collection name: ", collection_name)
        try:
            collection = chroma_client.create_collection(
                collection_name,
                embedding_function=sentence_transformer_ef,
            )
        except UniqueConstraintError:
            print("UniqueConstraintError error occured - making new collection")
            chroma_client.delete_collection(collection_name)
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef,
            )
            print("Deleted data from collection")
        print("Added data to collection")
        # print(self.actions)
        ids = [str(i) for i in range(len(self.actions))]
        collection.add(documents=[json.dumps(i) for i in self.actions], ids=ids)
        return "done"
