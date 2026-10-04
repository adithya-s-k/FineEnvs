from scripts import deploy_spaces


def test_upload_manifest_excludes_local_secrets_and_symlinks(tmp_path, monkeypatch):
    for name in ('app/main.py','web/app.css','.env','.env.production','sub/.env.local','sub/private.key','.local-data/run.json','.venv/env.py'):
        path=tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('sample')
    (tmp_path/'linked.py').symlink_to(tmp_path/'.env')
    (tmp_path/'linked-dir').symlink_to(tmp_path/'.local-data',target_is_directory=True)
    monkeypatch.setattr(deploy_spaces,'ROOT',tmp_path)
    assert deploy_spaces.files()==['app/main.py','web/app.css']


def test_admin_source_never_enters_public_deployment(tmp_path, monkeypatch):
    paths=['app/main.py','app/settings.py','app/admin.py','app/admin_app.py','web-admin/admin.js','web-admin/admin.css',
           'tests/test_security.py','scripts/deploy_spaces.py','web/js/main.js']
    for name in paths:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('sample')
    monkeypatch.setattr(deploy_spaces,'ROOT',tmp_path)
    assert deploy_spaces.files()==['app/main.py','app/settings.py','web/js/main.js']
    assert set(deploy_spaces.SPACES)=={'explorer'}
    assert deploy_spaces.SPACES['explorer']['visibility']=='public'
