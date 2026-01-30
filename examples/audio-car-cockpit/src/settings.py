from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class PydanticSettings(BaseSettings, case_sensitive=False):
    model_config = SettingsConfigDict(extra="ignore")

    DEMO_URL: HttpUrl
    AUDIO_SERVER_PORT: int


p_env = PydanticSettings()  # type:ignore[reportCallIssue]
