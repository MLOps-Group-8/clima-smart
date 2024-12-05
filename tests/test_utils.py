from unittest import TestCase, mock
from dags.utils import read_data_from_gcs, save_data_to_gcs, save_plot_to_gcs, save_object_to_gcs, load_object_from_gcs
import pandas as pd

class TestUtils(TestCase):

    @mock.patch('utils.storage.Client')
    def test_read_data_from_gcs(self, mock_client):
        # Mock GCS response
        mock_blob = mock.MagicMock()
        mock_blob.download_as_text.return_value = "col1,col2\n1,2\n3,4"
        mock_bucket = mock.MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client.return_value.bucket.return_value = mock_bucket

        df = read_data_from_gcs('test-bucket', 'test-blob.csv')
        self.assertEqual(df.shape, (2, 2))
        self.assertIn('col1', df.columns)

    @mock.patch('utils.storage.Client')
    def test_save_data_to_gcs(self, mock_client):
        # Mock GCS save functionality
        mock_bucket = mock.MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket

        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        save_data_to_gcs(df, 'test-bucket', 'test-file.csv')
        mock_bucket.blob.assert_called_with('test-file.csv')
    
    @mock.patch('utils.storage.Client')
    def test_save_plot_to_gcs(self, mock_client):
        mock_bucket = mock.MagicMock()
        mock_blob = mock.MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client.return_value.bucket.return_value = mock_bucket

        with mock.patch('matplotlib.pyplot.savefig'):
            save_plot_to_gcs('test-bucket', 'test-plot')
            mock_blob.upload_from_file.assert_called_once()

    @mock.patch('utils.storage.Client')
    def test_save_object_to_gcs(self, mock_client):
        mock_bucket = mock.MagicMock()
        mock_blob = mock.MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client.return_value.bucket.return_value = mock_bucket

        save_object_to_gcs('test-bucket', {'key': 'value'}, 'test.pkl')
        mock_blob.upload_from_file.assert_called_once()

    @mock.patch('utils.storage.Client')
    def test_load_object_from_gcs(self, mock_client):
        mock_bucket = mock.MagicMock()
        mock_blob = mock.MagicMock()
        mock_blob.download_as_string.return_value = pickle.dumps({'key': 'value'})
        mock_bucket.blob.return_value = mock_blob
        mock_client.return_value.bucket.return_value = mock_bucket

        result = load_object_from_gcs('test-bucket', 'test.pkl')
        self.assertEqual(result, {'key': 'value'})
