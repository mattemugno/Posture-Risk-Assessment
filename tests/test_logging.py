import io
import logging
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from src.utils import set_logger


class LoggingTests(unittest.TestCase):
    @contextmanager
    def configured(self, directory, level='INFO', existing_handler=None):
        logger = logging.Logger('logging-test')
        output = io.StringIO()
        if existing_handler is not None:
            logger.addHandler(existing_handler)
        try:
            with patch('src.utils.base_dir', directory), \
                    patch('src.utils.logging.getLogger', return_value=logger), \
                    patch('src.utils.sys.stdout', output), \
                    patch.dict(os.environ, {'LOG_LEVEL': level}):
                set_logger()
                yield logger, output
        finally:
            for handler in logger.handlers:
                handler.close()

    def test_repeated_setup_and_restart_preserve_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.configured(directory) as (logger, output):
                logger.info('first-message')
                set_logger()
                logger.info('second-message')
                logger.debug('hidden-debug-message')
                self.assertEqual(output.getvalue().count('first-message'), 1)
                self.assertEqual(output.getvalue().count('second-message'), 1)
                self.assertNotIn('hidden-debug-message', output.getvalue())
            with self.configured(directory) as (logger, _):
                logger.info('after-restart')
            text = Path(directory, 'logs', 'app.log').read_text()
            for message in ['first-message', 'second-message', 'after-restart']:
                self.assertEqual(text.count(message), 1)
            self.assertNotIn('hidden-debug-message', text)
            self.assertIn('[INFO]', text)

    def test_existing_console_handler_is_preserved_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            existing_output = io.StringIO()
            handler = logging.StreamHandler(existing_output)
            with self.configured(directory, existing_handler=handler) as (logger, output):
                logger.warning('single-warning')
                self.assertIn(handler, logger.handlers)
                self.assertEqual(existing_output.getvalue().count('single-warning'), 1)
                self.assertEqual(output.getvalue(), '')
            self.assertIn('single-warning', Path(directory, 'logs', 'app.log').read_text())

    def test_debug_level_can_be_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.configured(directory, level='debug') as (logger, output):
                logger.debug('debug-enabled')
                self.assertIn('debug-enabled', output.getvalue())
            self.assertIn('debug-enabled', Path(directory, 'logs', 'app.log').read_text())

    def test_invalid_level_falls_back_to_info(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.configured(directory, level='invalid') as (logger, output):
                self.assertEqual(logger.level, logging.INFO)
                self.assertIn('Unknown LOG_LEVEL', output.getvalue())


if __name__ == '__main__':
    unittest.main()
