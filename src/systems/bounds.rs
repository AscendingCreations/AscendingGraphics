use crate::Vec2;

/// View Bounds
/// ::::Used For::::
/// Clipping Text, Within Text internally.
/// Clipping objects, Using Rendering Scissor.
/// Checking Coords.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Bounds {
    pub left: f32,
    pub bottom: f32,
    pub right: f32,
    pub top: f32,
}

impl Bounds {
    /// Used to create [`Bounds`].
    ///
    /// # Arguments
    /// - left: Position from the left side of the screen.
    /// - bottom: Position from the bottom of the screen.
    /// - right: Position from the left side + Right offset.
    /// - top: Position from the bottom of the screen + top offset.
    ///
    pub fn new(left: f32, bottom: f32, right: f32, top: f32) -> Self {
        Self {
            left,
            bottom,
            right,
            top,
        }
    }

    /// Used to update offset x and y within a limited range.
    ///
    /// # Arguments
    /// - offset: Variable to Set and check against Self [`Bounds`] and limit [`Bounds`].
    /// - limits: Limit of what we will allow for lower or higher offset changes.
    ///
    pub fn set_offset_within_limits(&self, offset: &mut Vec2, limits: &Bounds) {
        if self.left + offset.x < limits.left {
            offset.x = limits.left - self.left;
        } else if self.right + offset.x > limits.right {
            offset.x = limits.right - self.right;
        }

        if self.bottom + offset.y < limits.bottom {
            offset.y = limits.bottom - self.bottom;
        } else if self.top + offset.y > limits.top {
            offset.y = limits.top - self.top;
        }
    }

    /// Used to add offset to [`Bounds`].
    ///
    /// # Arguments
    /// - offset: Amount to move the [`Bounds`].
    ///
    pub fn add_offset(&mut self, offset: Vec2) {
        self.left += offset.x;
        self.right += offset.x;
        self.top += offset.y;
        self.bottom += offset.y;
    }

    /// Used to adjust [`Bounds`] to a limited range.
    ///
    /// # Arguments
    /// - limits: limits to limit the [`Bounds`] too.
    ///
    pub fn set_within_limits(&mut self, limits: &Bounds) {
        if self.left < limits.left {
            self.left = limits.left;
        }

        if self.bottom < limits.bottom {
            self.bottom = limits.bottom;
        }

        if self.top > limits.top {
            self.top = limits.top;
        }

        if self.right > limits.right {
            self.right = limits.right;
        }
    }
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            left: 0.0,
            bottom: 0.0,
            right: 2_147_483_600.0,
            top: 2_147_483_600.0,
        }
    }
}
