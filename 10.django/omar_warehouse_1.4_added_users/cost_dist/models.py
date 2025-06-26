from django.db import models


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.BooleanField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.BooleanField()
    is_active = models.BooleanField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class CostDist(models.Model):
    trs_id = models.FloatField(blank=True, null=True)
    transaction_source = models.CharField(max_length=255, blank=True, null=True)
    project_no = models.FloatField(blank=True, null=True)
    project_name = models.CharField(max_length=255, blank=True, null=True)
    project_zone = models.CharField(max_length=255, blank=True, null=True)
    task_no = models.CharField(max_length=255, blank=True, null=True)
    task_name = models.CharField(max_length=255, blank=True, null=True)
    top_task_no = models.FloatField(blank=True, null=True)
    top_task_name = models.CharField(max_length=255, blank=True, null=True)
    po_no = models.CharField(max_length=255, blank=True, null=True)
    gl_date = models.DateTimeField(blank=True, null=True)
    expenditure_type = models.CharField(max_length=255, blank=True, null=True)
    project_location = models.CharField(max_length=255, blank=True, null=True)
    project_floor = models.CharField(max_length=255, blank=True, null=True)
    project_area = models.CharField(max_length=255, blank=True, null=True)
    expenditure_category = models.CharField(max_length=255, blank=True, null=True)
    expend_org = models.CharField(max_length=255, blank=True, null=True)
    amount = models.FloatField(blank=True, null=True)
    line_no = models.FloatField(blank=True, null=True)
    line_desc = models.CharField(max_length=255, blank=True, null=True)
    inv_no = models.CharField(max_length=255, blank=True, null=True)
    unit = models.CharField(max_length=255, blank=True, null=True)
    qty = models.FloatField(blank=True, null=True)
    ipc_no = models.FloatField(blank=True, null=True)
    supplier_no = models.CharField(max_length=255, blank=True, null=True)
    supplier_name = models.CharField(max_length=255, blank=True, null=True)
    supplier_site = models.CharField(max_length=255, blank=True, null=True)
    comment = models.CharField(max_length=255, blank=True, null=True)
    inventory_item = models.CharField(max_length=255, blank=True, null=True)
    owner = models.CharField(max_length=255, blank=True, null=True)
    distributions_status = models.CharField(max_length=255, blank=True, null=True)
    distributions_date = models.DateTimeField(blank=True, null=True)
    distributions_details = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'cost_dist'


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.SmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class MatMov(models.Model):
    trx_id = models.FloatField(blank=True, null=True)
    item_code = models.CharField(max_length=255, blank=True, null=True)
    item_desc = models.CharField(max_length=255, blank=True, null=True)
    unit = models.CharField(max_length=255, blank=True, null=True)
    from_field = models.CharField(db_column='from', max_length=255, blank=True, null=True)  # Field renamed because it was a Python reserved word.
    to = models.CharField(max_length=255, blank=True, null=True)
    project_name = models.CharField(max_length=255, blank=True, null=True)
    project_no = models.FloatField(blank=True, null=True)
    exp_id = models.CharField(max_length=255, blank=True, null=True)
    task_no = models.CharField(max_length=255, blank=True, null=True)
    exp_type = models.CharField(max_length=255, blank=True, null=True)
    material_type = models.CharField(max_length=255, blank=True, null=True)
    supplier_name = models.CharField(max_length=255, blank=True, null=True)
    supplier_no = models.FloatField(blank=True, null=True)
    trx_type = models.CharField(max_length=255, blank=True, null=True)
    move_order_no = models.CharField(max_length=255, blank=True, null=True)
    date = models.DateTimeField(blank=True, null=True)
    rate = models.FloatField(blank=True, null=True)
    qty = models.FloatField(blank=True, null=True)
    po_no = models.FloatField(blank=True, null=True)
    amount = models.FloatField(blank=True, null=True)
    project_area = models.CharField(max_length=255, blank=True, null=True)
    issued_type = models.CharField(max_length=255, blank=True, null=True)
    comment = models.CharField(max_length=255, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'mat_mov'


class Sysdiagrams(models.Model):
    name = models.CharField(max_length=128)
    principal_id = models.IntegerField()
    diagram_id = models.AutoField(primary_key=True)
    version = models.IntegerField(blank=True, null=True)
    definition = models.BinaryField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'sysdiagrams'
        unique_together = (('principal_id', 'name'),)
