UNFAZED_SETTINGS = {
    "ROOT_URLCONF": "entry.routes",
    "INSTALLED_APPS": ["event"],
    "MIDDLEWARE": [
        "unfazed_sentry.middleware.common.SentryMiddleware",
    ],
    "LIFESPAN": ["event.mcp.EventMCPLifespan"],
}
