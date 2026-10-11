import importlib
import os
import unittest
from datetime import datetime, timedelta, timezone

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


class ServiceTokenTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = cls.private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode('utf-8')
        os.environ['AI_SERVICE_JWT_PUBLIC_KEY'] = public_key
        os.environ['AI_SERVICE_JWT_ISSUER'] = 'systemvet-api'
        os.environ['AI_SERVICE_JWT_AUDIENCE'] = 'systemvet-ai'

        import service_token
        cls.security = importlib.reload(service_token)

    def token(self, operation='diagnostico', **overrides):
        now = datetime.now(timezone.utc)
        claims = {
            'sub': '7',
            'iss': 'systemvet-api',
            'aud': 'systemvet-ai',
            'iat': now,
            'nbf': now - timedelta(seconds=1),
            'exp': now + timedelta(seconds=45),
            'jti': 'solicitud-1',
            'token_type': 'ai_service',
            'operation': operation,
            'companyId': 9,
            'mascotaId': 21,
        }
        claims.update(overrides)
        return jwt.encode(claims, self.private_key, algorithm='RS256')

    def test_rechaza_un_token_destinado_a_otra_operacion(self):
        with self.assertRaises(self.security.ForbiddenServiceOperation):
            self.security.decode_service_token(self.token('laboratorio'), 'diagnostico')

    def test_rechaza_un_token_vencido(self):
        expired = datetime.now(timezone.utc) - timedelta(seconds=1)
        with self.assertRaises(self.security.InvalidServiceToken):
            self.security.decode_service_token(self.token(exp=expired), 'diagnostico')

    def test_acepta_la_delegacion_firmada_con_alcance_completo(self):
        claims = self.security.decode_service_token(self.token(), 'diagnostico')

        self.assertEqual(9, claims['companyId'])
        self.assertEqual(21, claims['mascotaId'])
        self.assertEqual('diagnostico', claims['operation'])


if __name__ == '__main__':
    unittest.main()
