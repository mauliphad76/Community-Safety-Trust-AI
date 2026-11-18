"""
Unit tests for authentication module (modules/auth.py).
Tests input validation, bcrypt password hashing, and authentication logic.
"""

import unittest
from unittest.mock import MagicMock, patch
import bcrypt

from modules.auth import (
    register_user,
    authenticate_user,
    USERNAME_REGEX,
    EMAIL_REGEX,
)


class TestAuthModule(unittest.TestCase):

    def test_regex_validations(self):
        # Valid usernames
        self.assertTrue(bool(USERNAME_REGEX.match("john_doe")))
        self.assertTrue(bool(USERNAME_REGEX.match("alice123")))
        self.assertTrue(bool(USERNAME_REGEX.match("user_99")))

        # Invalid usernames
        self.assertFalse(bool(USERNAME_REGEX.match("ab")))  # too short
        self.assertFalse(bool(USERNAME_REGEX.match("john doe")))  # space
        self.assertFalse(bool(USERNAME_REGEX.match("john@doe")))  # special char

        # Valid emails
        self.assertTrue(bool(EMAIL_REGEX.match("user@example.com")))
        self.assertTrue(bool(EMAIL_REGEX.match("john.doe@sub.domain.org")))

        # Invalid emails
        self.assertFalse(bool(EMAIL_REGEX.match("invalid_email")))
        self.assertFalse(bool(EMAIL_REGEX.match("@example.com")))

    def test_registration_validation_errors(self):
        # Missing full name
        success, msg = register_user("", "johndoe", "john@example.com", "pass123", "pass123")
        self.assertFalse(success)
        self.assertIn("full name", msg.lower())

        # Password mismatch
        success, msg = register_user("John Doe", "johndoe", "john@example.com", "password123", "different")
        self.assertFalse(success)
        self.assertIn("do not match", msg.lower())

        # Short password
        success, msg = register_user("John Doe", "johndoe", "john@example.com", "123", "123")
        self.assertFalse(success)
        self.assertIn("at least 6 characters", msg.lower())

    @patch("modules.auth.get_mongo_uri", return_value="mongodb+srv://fake:fake@cluster0.mongodb.net")
    @patch("modules.auth.get_users_collection")
    def test_successful_registration_and_login(self, mock_get_users, mock_get_uri):
        # Mock MongoDB users collection
        mock_users = MagicMock()
        mock_get_users.return_value = mock_users

        # Simulate user does not exist yet
        mock_users.find_one.return_value = None

        success, msg = register_user("Alice Smith", "alicesmith", "alice@example.com", "secure123", "secure123")
        self.assertTrue(success)
        self.assertIn("successful", msg.lower())

        # Check insert_one was called with hashed password
        self.assertTrue(mock_users.insert_one.called)
        inserted_doc = dict(mock_users.insert_one.call_args[0][0])
        inserted_doc["_id"] = "mock_id_12345"
        self.assertEqual(inserted_doc["username"], "alicesmith")
        self.assertEqual(inserted_doc["email"], "alice@example.com")
        self.assertTrue(bcrypt.checkpw(b"secure123", inserted_doc["password_hash"].encode("utf-8")))

        # Now test authentication with the created hash
        mock_users.find_one.return_value = inserted_doc

        # Correct password
        login_success, login_msg, user_data = authenticate_user("alicesmith", "secure123")
        self.assertTrue(login_success)
        self.assertEqual(user_data["username"], "alicesmith")

        # Wrong password
        wrong_success, wrong_msg, wrong_user = authenticate_user("alicesmith", "wrongpass")
        self.assertFalse(wrong_success)
        self.assertIsNone(wrong_user)


if __name__ == "__main__":
    unittest.main()
