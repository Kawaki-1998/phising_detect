from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from src.config.config import CASSANDRA_CONFIG
import logging

logger = logging.getLogger(__name__)

class CassandraClient:
    def __init__(self):
        self.cluster = None
        self.session = None
        self.keyspace = CASSANDRA_CONFIG["keyspace"]

    def connect(self):
        try:
            self.cluster = Cluster(
                [CASSANDRA_CONFIG["host"]],
                port=CASSANDRA_CONFIG["port"]
            )
            self.session = self.cluster.connect()
            self._create_keyspace()
            self._create_tables()
            logger.info("Successfully connected to Cassandra cluster")
        except Exception as e:
            logger.error(f"Error connecting to Cassandra cluster: {str(e)}")
            raise

    def _create_keyspace(self):
        self.session.execute("""
            CREATE KEYSPACE IF NOT EXISTS phishing_detection
            WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}
        """)
        self.session.set_keyspace(self.keyspace)

    def _create_tables(self):
        self.session.execute("""
            CREATE TABLE IF NOT EXISTS domain_logs (
                domain text PRIMARY KEY,
                prediction boolean,
                confidence float,
                features map<text, float>,
                timestamp timestamp,
                model_version text
            )
        """)

    def insert_prediction(self, domain, prediction, confidence, features, model_version):
        try:
            query = """
                INSERT INTO domain_logs (domain, prediction, confidence, features, timestamp, model_version)
                VALUES (%s, %s, %s, %s, toTimestamp(now()), %s)
            """
            self.session.execute(query, (domain, prediction, confidence, features, model_version))
            return True
        except Exception as e:
            logger.error(f"Error inserting prediction: {str(e)}")
            return False

    def get_prediction(self, domain):
        try:
            query = "SELECT * FROM domain_logs WHERE domain = %s"
            result = self.session.execute(query, (domain,)).one()
            return result
        except Exception as e:
            logger.error(f"Error retrieving prediction: {str(e)}")
            return None

    def close(self):
        if self.cluster:
            self.cluster.shutdown()
            logger.info("Cassandra connection closed") 