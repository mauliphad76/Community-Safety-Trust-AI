import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StreamlitIntegrationTests(unittest.TestCase):
    def create_app(self, authenticated=True):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120)
        if authenticated:
            app.session_state["authenticated"] = True
            app.session_state["user"] = {
                "username": "testuser",
                "full_name": "Test User",
                "email": "test@example.com",
            }
        app.run()
        self.assertFalse(
            app.exception,
            [exception.message for exception in app.exception],
        )
        return app

    def test_unauthenticated_user_sees_login_and_register_flow(self):
        app = self.create_app(authenticated=False)
        # Should render login and register input keys
        input_keys = {item.key for item in app.text_input if item.key}
        self.assertIn("login_user_input", input_keys)
        self.assertIn("login_pass_input", input_keys)
        self.assertIn("reg_username_input", input_keys)
        self.assertIn("reg_email_input", input_keys)

    def test_dashboard_navigation_opens_all_three_modules(self):
        app = self.create_app()
        self.assertTrue(any(button.key == "open_currency" for button in app.button))
        self.assertTrue(any(button.key == "open_activity" for button in app.button))
        self.assertTrue(any(button.key == "open_scam" for button in app.button))

        app.button(key="navigate_currency").click().run()
        self.assertFalse(app.exception, [item.message for item in app.exception])
        self.assertIsNotNone(app.file_uploader(key="currency_upload_ui"))

        app.button(key="navigate_activity").click().run()
        self.assertFalse(app.exception, [item.message for item in app.exception])
        self.assertIsNotNone(app.file_uploader(key="activity_image_upload"))

        app.button(key="navigate_scam").click().run()
        self.assertFalse(app.exception, [item.message for item in app.exception])
        self.assertIsNotNone(app.text_area(key="fake_news_text"))

    def test_fake_news_ui_shows_a_score_for_english_and_hindi(self):
        app = self.create_app()
        app.button(key="navigate_scam").click().run()

        samples = (
            (
                "English",
                "The city council approved its annual budget after a public hearing.",
            ),
            (
                "Hindi",
                "नगर परिषद ने सार्वजनिक सुनवाई के बाद वार्षिक बजट को मंजूरी दी।",
            ),
        )
        for language, text in samples:
            with self.subTest(language=language):
                app.selectbox(key="fake_news_language").select(language).run()
                app.text_area(key="fake_news_text").set_value(text)
                app.button(key="fake_news_analyze").click().run()
                self.assertFalse(app.exception, [item.message for item in app.exception])
                self.assertTrue(
                    any(metric.label == "Signed decision score" for metric in app.metric)
                )
                self.assertTrue(app.success or app.error)
                self.assertTrue(
                    any(
                        "not a calibrated probability" in caption.value
                        for caption in app.caption
                    )
                )


if __name__ == "__main__":
    unittest.main()
