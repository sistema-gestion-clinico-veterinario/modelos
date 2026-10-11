import base64
import os
from functools import lru_cache

import jwt


class ServiceTokenConfigurationError(RuntimeError):
    pass


class InvalidServiceToken(ValueError):
    pass


class ForbiddenServiceOperation(ValueError):
    pass


@lru_cache(maxsize=1)
def public_key() -> str:
    """Carga la clave pública del backend sin incorporar secretos al repositorio."""
    encoded = os.getenv('AI_SERVICE_JWT_PUBLIC_KEY_BASE64', '').strip()
    if encoded:
        try:
            return base64.b64decode(encoded, validate=True).decode('utf-8')
        except (ValueError, UnicodeDecodeError) as exc:
            raise ServiceTokenConfigurationError(
                'AI_SERVICE_JWT_PUBLIC_KEY_BASE64 no es válida'
            ) from exc

    pem = os.getenv('AI_SERVICE_JWT_PUBLIC_KEY', '').replace('\\n', '\n').strip()
    if pem:
        return pem
    raise ServiceTokenConfigurationError(
        'La clave pública para autorizar al backend no está configurada'
    )


def decode_service_token(token: str, expected_operation: str) -> dict:
    try:
        claims = jwt.decode(
            token,
            public_key(),
            algorithms=['RS256'],
            issuer=os.getenv('AI_SERVICE_JWT_ISSUER', 'systemvet-api'),
            audience=os.getenv('AI_SERVICE_JWT_AUDIENCE', 'systemvet-ai'),
            options={
                'require': ['sub', 'iss', 'aud', 'exp', 'iat', 'nbf', 'jti',
                            'token_type', 'operation', 'companyId', 'mascotaId']
            },
        )
    except ServiceTokenConfigurationError:
        raise
    except jwt.PyJWTError as exc:
        raise InvalidServiceToken('Autorización interna inválida o vencida') from exc

    if claims.get('token_type') != 'ai_service' or claims.get('operation') != expected_operation:
        raise ForbiddenServiceOperation('Operación no autorizada')
    if not isinstance(claims.get('companyId'), int) or not isinstance(claims.get('mascotaId'), int):
        raise ForbiddenServiceOperation('Alcance de datos inválido')
    return claims
