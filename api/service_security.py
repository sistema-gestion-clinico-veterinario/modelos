from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from service_token import (
    ForbiddenServiceOperation,
    InvalidServiceToken,
    ServiceTokenConfigurationError,
    decode_service_token,
)


_bearer = HTTPBearer(auto_error=False)
def require_ai_operation(operation: str) -> Callable:
    """Exige una autorización delegada válida y limitada al endpoint solicitado."""

    async def verify(
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> dict:
        if credentials is None or credentials.scheme.lower() != 'bearer':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Autenticación interna requerida',
                headers={'WWW-Authenticate': 'Bearer'},
            )

        try:
            return decode_service_token(credentials.credentials, operation)
        except ServiceTokenConfigurationError as exc:
            raise HTTPException(status_code=503, detail='Seguridad interna no configurada') from exc
        except InvalidServiceToken as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Autorización interna inválida o vencida',
                headers={'WWW-Authenticate': 'Bearer'},
            ) from exc
        except ForbiddenServiceOperation as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    return verify
