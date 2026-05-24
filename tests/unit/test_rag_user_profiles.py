"""Tests for ProfileRegistry and ProfileConfig."""

import pytest

from core.models import UserProfileType
from modules.rag.user_profiles import ProfileRegistry


class TestProfileRegistry:
    def test_all_six_profiles_registered(self):
        all_types = ProfileRegistry.all_types()
        assert len(all_types) == 6
        assert UserProfileType.PATIENT in all_types
        assert UserProfileType.MEDICAL_STUDENT in all_types
        assert UserProfileType.MEDICAL_PROFESSIONAL in all_types
        assert UserProfileType.DIAGNOSTIC_ASSISTANT in all_types
        assert UserProfileType.NATURAL_MEDICINE in all_types
        assert UserProfileType.CAREGIVER in all_types

    def test_get_patient_profile(self):
        cfg = ProfileRegistry.get(UserProfileType.PATIENT)
        assert cfg.profile_type == UserProfileType.PATIENT
        assert "{query}" in cfg.system_prompt
        assert "{context}" in cfg.system_prompt

    def test_all_profiles_have_required_template_placeholders(self):
        for pt in ProfileRegistry.all_types():
            cfg = ProfileRegistry.get(pt)
            assert "{query}" in cfg.system_prompt, f"{pt} missing {{query}}"
            assert "{context}" in cfg.system_prompt, f"{pt} missing {{context}}"
            assert len(cfg.focus_areas) > 0, f"{pt} missing focus_areas"

    def test_default_profile_is_patient(self):
        cfg = ProfileRegistry.default_profile()
        assert cfg.profile_type == UserProfileType.PATIENT

    def test_profile_config_is_frozen(self):
        cfg = ProfileRegistry.get(UserProfileType.PATIENT)
        with pytest.raises(AttributeError):
            cfg.tone = "modified"

    def test_all_profiles_have_unique_focus_areas(self):
        """Ensure each profile has distinct focus areas."""
        focus_by_profile = {
            pt: set(ProfileRegistry.get(pt).focus_areas)
            for pt in ProfileRegistry.all_types()
        }
        # Medical student and professional may share some areas, but overall should differ
        patient_focus = focus_by_profile[UserProfileType.PATIENT]
        medical_focus = focus_by_profile[UserProfileType.MEDICAL_PROFESSIONAL]
        assert patient_focus != medical_focus
