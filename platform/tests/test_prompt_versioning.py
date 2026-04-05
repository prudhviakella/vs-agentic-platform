"""
tests/test_prompt_versioning.py — Prompt Versioning Unit Tests
"""

import pytest
from unittest.mock import MagicMock, patch


class TestPromptVersionManager:

    def test_activate_version_updates_ssm(self):
        from platform.prompt_versioning import manager

        with patch.object(manager, "_resolve_app_name", return_value="clinical-trial-agent"), \
             patch.object(manager, "get_active_version", return_value="2"),                   \
             patch.object(manager, "_validate_version_exists"),                                \
             patch.object(manager, "_put_ssm") as mock_put:

            previous, activated = manager.activate_version("clinical-trial", "prod", "3")

        assert previous  == "2"
        assert activated == "3"
        assert mock_put.call_count == 2  # previous + active

    def test_rollback_swaps_versions(self):
        from platform.prompt_versioning import manager

        with patch.object(manager, "_resolve_app_name", return_value="clinical-trial-agent"), \
             patch.object(manager, "get_active_version", return_value="3"),                   \
             patch.object(manager, "_get_ssm_optional",  return_value="2"),                   \
             patch.object(manager, "_put_ssm") as mock_put:

            rolled_from, rolled_to = manager.rollback_version("clinical-trial", "prod")

        assert rolled_from == "3"
        assert rolled_to   == "2"

    def test_rollback_raises_when_no_previous(self):
        from platform.prompt_versioning import manager

        with patch.object(manager, "_resolve_app_name", return_value="clinical-trial-agent"), \
             patch.object(manager, "get_active_version", return_value="1"),                   \
             patch.object(manager, "_get_ssm_optional",  return_value=None):

            with pytest.raises(ValueError, match="No previous version"):
                manager.rollback_version("clinical-trial", "prod")

    def test_unknown_agent_raises(self):
        from platform.prompt_versioning import manager
        with pytest.raises(ValueError, match="Unknown agent"):
            manager._resolve_app_name("unknown-agent")
